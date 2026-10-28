# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging
import sys
from typing import Optional

import torch
import triton
import triton.language as tl

from .fused_moe_kernel import (
    dispatch_kunlunxin_fused_moe_kernel,
    invoke_kunlunxin_fused_moe_kernel,
    invoke_kunlunxin_moe_sum,
)

logger = logging.getLogger(__name__)

# The public dispatch helper is imported before vendor overrides are applied and is
# not part of the operator registrar. Replace it while loading this registered
# implementation so direct calls use the same Kunlunxin Triton kernel.
setattr(
    sys.modules["flag_gems"],
    "dispatch_fused_moe_kernel",
    dispatch_kunlunxin_fused_moe_kernel,
)


@triton.jit
def _experts_route_cast_kernel(src, dst, count: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src + i, mask=i < count, other=0).to(tl.int32)
    tl.store(dst + i, value, mask=i < count)


@triton.jit
def _experts_swiglu_kernel(
    src, dst, width: tl.constexpr, count: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // width, i % width
    gate = tl.load(src + row * (2 * width) + col, mask=i < count, other=0).to(
        tl.float32
    )
    up = tl.load(src + row * (2 * width) + width + col, mask=i < count, other=0).to(
        tl.float32
    )
    value = (gate * tl.sigmoid(gate)) * up
    tl.store(dst + i, value, mask=i < count)


@triton.jit
def _experts_finish_kernel(
    src,
    weights,
    dst,
    width: tl.constexpr,
    count: tl.constexpr,
    TOPK: tl.constexpr,
    WEIGHTED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // width, i % width
    acc = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(TOPK):
        value = tl.load(
            src + (row * TOPK + k) * width + col, mask=i < count, other=0
        ).to(tl.float32)
        if WEIGHTED:
            weight = tl.load(weights + row * TOPK + k, mask=i < count, other=0).to(
                tl.float32
            )
            value = (value * weight).to(src.dtype.element_ty).to(tl.float32)
        acc += value
    tl.store(dst + i, acc, mask=i < count)


# (data_ptr, numel) -> version of the routing tensor at the previous call.
_VENDOR_ROUTE_SIG: dict = {}


def _vendor_route_content_unchanged(topk_ids: torch.Tensor) -> bool:
    """True when the routing content at this address is unchanged since the
    previous call. Uses the tensor's autograd version counter, so this costs
    no device sync and no allocation.
    """
    key = (topk_ids.data_ptr(), topk_ids.numel())
    version = topk_ids._version
    previous = _VENDOR_ROUTE_SIG.get(key)
    _VENDOR_ROUTE_SIG[key] = version
    return previous is None or previous == version


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN FUSED_EXPERTS_IMPL")
    if activation != "silu":
        raise NotImplementedError("Kunlunxin fused_experts_impl supports SiLU only")

    unsupported_quantization = any(
        (
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            ocp_mx_scheme is not None,
            per_channel_quant,
            w1_scale is not None,
            w2_scale is not None,
            w1_zp is not None,
            w2_zp is not None,
            a1_scale is not None,
            a2_scale is not None,
            block_shape is not None,
        )
    )
    if unsupported_quantization:
        raise NotImplementedError(
            "Kunlunxin fused_experts_impl does not support quantized weights"
        )
    if expert_map is not None:
        raise NotImplementedError(
            "Kunlunxin fused_experts_impl does not support expert parallelism"
        )

    num_tokens, hidden_size = hidden_states.shape
    num_experts, gate_up_size, w1_hidden_size = w1.shape
    top_k = topk_ids.size(1)
    intermediate_size = gate_up_size // 2
    configured_num_experts = (
        num_experts if global_num_experts == -1 else global_num_experts
    )

    assert configured_num_experts == num_experts
    assert hidden_states.is_contiguous()
    assert w1.stride(-1) == 1 and w2.stride(-1) == 1
    assert topk_weights.is_contiguous() and topk_ids.is_contiguous()
    assert topk_weights.shape == topk_ids.shape == (num_tokens, top_k)
    assert gate_up_size % 2 == 0
    assert w1_hidden_size == hidden_size
    assert w2.shape == (num_experts, hidden_size, intermediate_size)
    assert hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert w1.dtype == hidden_states.dtype and w2.dtype == hidden_states.dtype

    config = {"BLOCK_SIZE_M": 16}
    # Leave the fast path only when a reused routing address carries changed
    # content since the previous call.
    generic_bf16_route = not _vendor_route_content_unchanged(topk_ids)
    # Convert routing metadata on the producer stream, without native casts.
    if topk_ids.dtype != torch.int32:
        route_ids = torch.empty(
            topk_ids.shape, dtype=torch.int32, device=topk_ids.device
        )
        _experts_route_cast_kernel[(triton.cdiv(topk_ids.numel(), 256),)](
            topk_ids,
            route_ids,
            topk_ids.numel(),
            BLOCK=256,
            num_warps=1,
            num_stages=1,
        )
        topk_ids = route_ids

    use_xblas_struct = (
        not apply_router_weight_on_input and w1_bias is None and w2_bias is None
    )
    if use_xblas_struct:
        # Plain GEMMs with Triton-only epilogues.
        intermediate13 = torch.empty(
            (num_tokens, top_k, gate_up_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        invoke_kunlunxin_fused_moe_kernel(
            hidden_states,
            w1,
            intermediate13,
            None,
            None,
            topk_ids,
            topk_ids,
            topk_ids,
            False,
            top_k,
            config,
            FUSE_SILU=False,
            direct_routing=True,
            generic_route=generic_bf16_route,
        )
        intermediate = torch.empty(
            (num_tokens, top_k, intermediate_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        _experts_swiglu_kernel[(triton.cdiv(intermediate.numel(), 256),)](
            intermediate13,
            intermediate,
            intermediate_size,
            intermediate.numel(),
            BLOCK=256,
            num_warps=1,
            num_stages=1,
        )
    else:
        intermediate = torch.empty(
            (num_tokens, top_k, intermediate_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        invoke_kunlunxin_fused_moe_kernel(
            hidden_states,
            w1,
            intermediate,
            w1_bias,
            topk_weights if apply_router_weight_on_input else None,
            topk_ids,
            topk_ids,
            topk_ids,
            False,
            top_k,
            config,
            FUSE_SILU=True,
            direct_routing=True,
            routed_weight_on_input=apply_router_weight_on_input,
            generic_route=generic_bf16_route,
        )

    routed_output = torch.empty(
        (num_tokens, top_k, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    invoke_kunlunxin_fused_moe_kernel(
        intermediate.view(num_tokens * top_k, intermediate_size),
        w2,
        routed_output,
        w2_bias,
        (
            None
            if use_xblas_struct
            else None if apply_router_weight_on_input else topk_weights
        ),
        topk_ids,
        topk_ids,
        topk_ids,
        (False if use_xblas_struct else not apply_router_weight_on_input),
        1,
        config,
        FUSE_SILU=False,
        direct_routing=True,
        generic_route=generic_bf16_route,
    )
    # All reads of hidden_states have completed before this final stream-ordered
    # reduction, so the in-place variant can safely target the original storage.
    output = hidden_states if inplace else torch.empty_like(hidden_states)
    _experts_finish_kernel[(triton.cdiv(output.numel(), 256),)](
        routed_output,
        topk_weights,
        output,
        hidden_size,
        output.numel(),
        TOPK=top_k,
        WEIGHTED=use_xblas_struct,
        BLOCK=256,
        num_warps=1,
        num_stages=1,
    )
    return output


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> None:
    """Write the final Triton reduction directly into hidden_states."""
    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=True,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )


def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )


# Route the flag_gems-level MoE entry points to this implementation as well.
# Runs at import time AFTER the defs above; the dispatch hijack at the top of
# this file already replaced flag_gems.dispatch_fused_moe_kernel.
_fg_mod = sys.modules["flag_gems"]
_fg_mod.fused_experts_impl = fused_experts_impl
_fg_mod.inplace_fused_experts = inplace_fused_experts
_fg_mod.outplace_fused_experts = outplace_fused_experts
_fg_mod.moe_sum = invoke_kunlunxin_moe_sum
