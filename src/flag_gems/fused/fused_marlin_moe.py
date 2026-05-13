# SPDX-License-Identifier: Apache-2.0
"""
Fused Marlin MoE for FlagGems.

Aligns the interface of vLLM v0.20.0:
    vllm/model_executor/layers/fused_moe/fused_marlin_moe.py :: fused_marlin_moe

PHASE 2 (this file): bypass `fused_experts_impl`'s dequant-then-FP16-GEMM
shortcut and dispatch directly to the wna16 Triton kernel
(`fused_moe_kernel_gptq_awq`) for true fused-dequant W4A16/W8A16 GEMM.

The local helper `_fused_marlin_moe_impl` mirrors `fused_experts_impl`'s
orchestration (chunk loop, moe_align, two GEMMs, activation, reduction)
but deletes the INT4/INT8 dequant branch and forwards `block_shape` so
the wna16 path is actually taken.

MVP scope:
  - quant_type: GPTQ uint4b8 (INT4) and uint8b128 (INT8)
  - activation: SwiGLU / SiLU
  - act_order:  NOT supported (g_idx / sort_indices must be None)
  - FP8 input:  NOT supported
  - LoRA, clamp_limit, expert_map: NOT supported
"""
from typing import Any, Callable, List, Optional

import torch
import triton.language as tl

from flag_gems.fused.fused_moe import (
    MoEActivation,
    _get_config_dtype_str,
    _get_config_quant_dtype,
    apply_moe_activation,
    dispatch_fused_moe_kernel,
    moe_kernel_quantize_input,
    try_get_optimal_moe_config,
)
from flag_gems.fused.moe_align_block_size import moe_align_block_size
from flag_gems.fused.moe_sum import moe_sum

import functools


# ----------------------------------------------------------------------------
# quant_type_id constants — mirror a subset of vLLM scalar_types ids.
# ----------------------------------------------------------------------------
# GPTQ INT4 (weight stored as w + 8, dequant subtracts 8)
QUANT_TYPE_UINT4B8 = 0
# INT8 (weight stored as w + 128)
QUANT_TYPE_UINT8B128 = 1

_QUANT_TYPE_INT4 = {QUANT_TYPE_UINT4B8}
_QUANT_TYPE_INT8 = {QUANT_TYPE_UINT8B128}
_SUPPORTED_QUANT_TYPES = _QUANT_TYPE_INT4 | _QUANT_TYPE_INT8


# ----------------------------------------------------------------------------
# Phase-2 impl: copy of fused_experts_impl but with the dequant shortcut
# removed so the wna16 Triton kernel is actually invoked for W4A16/W8A16.
# ----------------------------------------------------------------------------
def _fused_marlin_moe_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Like fused_experts_impl, but:
      - drops all paths irrelevant to W4A16/W8A16 (no fp8, int8_w8a8, mxfp).
      - REMOVES the `w = w.to(fp16) * scale.unsqueeze(-1)` dequant shortcut.
      - forwards block_shape so the wna16 kernel uses the right group_size.
    """
    assert (
        activation == "silu"
    ), f"Only 'silu' activation is supported, got {activation}"
    assert use_int4_w4a16 or use_int8_w8a16, (
        "_fused_marlin_moe_impl expects a quantized path"
    )

    activation_enum = MoEActivation.from_str(activation)

    # Packed-aware shape check.
    # W4A16 (pack_factor=2): w1.size(2) == K // 2
    # W8A16 (pack_factor=1): w1.size(2) == K
    expected_packed_k = (
        hidden_states.size(1) // 2 if use_int4_w4a16 else hidden_states.size(1)
    )
    assert w1.size(2) == expected_packed_k, (
        f"w1 packed K mismatch: hidden_size={hidden_states.size(1)}, "
        f"use_int4_w4a16={use_int4_w4a16}, expected w1.size(2)={expected_packed_k}, "
        f"got {w1.size(2)}"
    )

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)

    CHUNK_SIZE: int = 16 * 1024
    M = min(num_tokens, CHUNK_SIZE)

    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=False,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=None,
        dtype=hidden_states.dtype,
    )
    quant_dtype = _get_config_quant_dtype(
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        ocp_mx_scheme=None,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        block_shape=block_shape,
        E=E,
    )
    config = get_config_func(M)
    config["SPLIT_K"] = 1

    # cache1 and cache3 share memory (non-overlapping lifetime)
    cache13 = torch.empty(
        M * top_k_num * max(N, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)

    activation_out_dim = MoEActivation.adjust_N_for_activation(N, activation_enum)
    intermediate_cache2 = torch.empty(
        (M * top_k_num, activation_out_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    out_hidden_states = hidden_states if inplace else torch.empty_like(hidden_states)

    # ★ Phase-2 KEY DIFFERENCE: the W4A16/W8A16 dequant shortcut that lived
    # here in `fused_experts_impl` is intentionally REMOVED. The wna16
    # Triton kernel will consume INT4 weights + scale directly.

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.size()

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[
                : tokens_in_chunk * topk_ids.size(1)
            ]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)
            config["SPLIT_K"] = 1

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        # Activation quantization is a no-op for W4A16/W8A16 (no input quant).
        qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
            A=curr_hidden_states,
            A_scale=None,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
            ocp_mx_scheme=None,
        )

        # Use the routed-path (skip the SPARSITY_FACTOR shortcut, which is
        # explicitly disabled for quantized + block_shape configs anyway).
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids,
            config["BLOCK_SIZE_M"],
            global_num_experts,
            expert_map,
        )

        # ----- GEMM 1: hidden @ w1  (fused dequant on B inside the kernel) -----
        dispatch_fused_moe_kernel(
            qcurr_hidden_states,
            w1,
            intermediate_cache1,
            a1q_scale,
            w1_scale,
            w1_zp,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            top_k_num,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w1_bias,
        )

        # ----- Activation: SwiGLU = silu(gate) * up -----
        apply_moe_activation(
            activation_enum, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            A=intermediate_cache2,
            A_scale=None,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
            ocp_mx_scheme=None,
        )

        if expert_map is not None:
            intermediate_cache3.zero_()

        # ----- GEMM 2: act @ w2  (fused dequant on B inside the kernel) -----
        dispatch_fused_moe_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            w2_scale,
            w2_zp,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w2_bias,
        )

        # ----- Reduce: sum topk-weighted expert outputs back per token -----
        moe_sum(
            intermediate_cache3.view(*intermediate_cache3.size()),
            out_hidden_states[begin_chunk_idx:end_chunk_idx],
        )

    return out_hidden_states


# ----------------------------------------------------------------------------
# Public entry point: vLLM-aligned wrapper.
# ----------------------------------------------------------------------------
def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: Any = None,
    activation_func: Optional[Callable] = None,
    moe_sum: Optional[Callable] = None,
    expert_map: Optional[torch.Tensor] = None,
    input_global_scale1: Optional[torch.Tensor] = None,
    input_global_scale2: Optional[torch.Tensor] = None,
    global_scale1: Optional[torch.Tensor] = None,
    global_scale2: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    intermediate_cache13: Optional[torch.Tensor] = None,
    intermediate_cache2: Optional[torch.Tensor] = None,
    is_k_full: bool = True,
    output: Optional[torch.Tensor] = None,
    input_dtype: Optional[torch.dtype] = None,
    inplace: bool = False,
    clamp_limit: Optional[float] = None,
    group_size: int = 128,
) -> torch.Tensor:
    """Phase-2 entry point: dispatch to local wna16-using impl."""
    # ---- MVP guardrails --------------------------------------------------
    if quant_type_id not in _SUPPORTED_QUANT_TYPES:
        raise NotImplementedError(
            f"MVP supports quant_type_id in {_SUPPORTED_QUANT_TYPES}, "
            f"got {quant_type_id}"
        )
    if g_idx1 is not None or g_idx2 is not None:
        raise NotImplementedError("act_order (g_idx) not yet supported in MVP")
    if sort_indices1 is not None or sort_indices2 is not None:
        raise NotImplementedError(
            "act_order (sort_indices) not yet supported in MVP"
        )
    if input_dtype is not None:
        raise NotImplementedError("FP8 / INT8 input quantization not supported")
    if clamp_limit is not None:
        raise NotImplementedError("clamp_limit (GLM-4 swiglu) not supported")
    if input_global_scale1 is not None or input_global_scale2 is not None:
        raise NotImplementedError("input_global_scale not supported in MVP")
    if global_scale1 is not None or global_scale2 is not None:
        raise NotImplementedError("global_scale not supported in MVP")

    use_int4_w4a16 = quant_type_id in _QUANT_TYPE_INT4
    use_int8_w8a16 = quant_type_id in _QUANT_TYPE_INT8

    activation_str = "silu"
    if activation is not None:
        for attr in ("value", "name"):
            v = getattr(activation, attr, None)
            if isinstance(v, str):
                activation_str = v.lower()
                break
        if isinstance(activation, str):
            activation_str = activation.lower()
    if activation_str != "silu":
        raise NotImplementedError(
            f"MVP only supports SiLU/SwiGLU activation, got {activation_str}"
        )

    if inplace and output is not None:
        raise ValueError("Cannot pass both inplace=True and output")

    result = _fused_marlin_moe_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation=activation_str,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zeros,
        w2_zp=w2_zeros,
        w1_bias=bias1,
        w2_bias=bias2,
        # Critical for Phase 2: block_shape=[0, group_size] makes the
        # wna16 Triton kernel use the per-group scales correctly.
        block_shape=[0, group_size],
    )

    if output is not None:
        output.copy_(result)
        return output
    return result


__all__ = ["fused_marlin_moe", "QUANT_TYPE_UINT4B8", "QUANT_TYPE_UINT8B128"]
