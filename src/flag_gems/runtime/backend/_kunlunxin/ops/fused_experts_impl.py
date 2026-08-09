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
    logger.debug("GEMS_KUNLUNXIN FUSED EXPERTS IMPL")
    if inplace:
        raise NotImplementedError(
            "Kunlunxin fused_experts_impl supports out-of-place only"
        )
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
        None if apply_router_weight_on_input else topk_weights,
        topk_ids,
        topk_ids,
        topk_ids,
        not apply_router_weight_on_input,
        1,
        config,
        FUSE_SILU=False,
        direct_routing=True,
    )

    output = torch.empty_like(hidden_states)
    invoke_kunlunxin_moe_sum(routed_output, output)
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
    """In-place fused MoE: writes the routed result back into ``hidden_states``.

    Uses the Kunlunxin out-of-place Triton kernels and writes the result back
    with the native strided copy (``aten::_copy_from``) so the registered
    ``copy_`` override is not re-entered.
    """
    output = fused_experts_impl(
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
    torch.ops.aten._copy_from(output, hidden_states, False)


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
