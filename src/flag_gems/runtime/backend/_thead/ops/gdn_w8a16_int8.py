# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
import triton
import triton.language as tl

from flag_gems.fused.FLA.triton_ops_helper import exp
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def store_int8_state(values, state, scales, offsets, scale_offsets, mask, mask_v):
    magnitude = tl.max(tl.abs(values), axis=1)
    safe_magnitude = tl.where(magnitude > 0, magnitude, 127.0)
    scale = safe_magnitude * (1.0 / 127.0)
    reciprocal_scale = 127.0 / safe_magnitude
    # Share one division across the channel instead of dividing every element.
    quantized = tl.minimum(
        tl.maximum(tl.floor(values * reciprocal_scale[:, None] + 0.5), -127), 127
    )
    tl.store(state + offsets, quantized.to(tl.int8), mask)
    tl.store(scales + scale_offsets, scale, mask_v)


@libentry()
@triton.jit
def quantize_state_kernel(
    source,
    state,
    scales,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    group = tl.program_id(1).to(tl.int64)
    column = tl.program_id(0) * BV + tl.arange(0, BV)
    reduction = tl.arange(0, BK)
    offsets = (group * V + column[:, None]) * K + reduction[None, :]
    mask = (column[:, None] < V) & (reduction[None, :] < K)
    values = tl.load(source + offsets, mask, 0).to(tl.float32)
    store_int8_state(
        values, state, scales, offsets, group * V + column, mask, column < V
    )


def quantize_gdn_state_int8(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize contiguous [slots, value_heads, V, K] state per V channel."""
    logger.debug("GEMS_THEAD QUANTIZE_GDN_STATE_INT8")
    if state.ndim != 4 or not state.is_contiguous():
        raise ValueError("state must be contiguous [slots, value_heads, V, K]")
    if state.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("state must be floating point")
    if state.device.type != "cuda" or state.requires_grad:
        raise ValueError("state must be a forward-only accelerator tensor")
    slots, heads, columns, reduction = state.shape
    if reduction < 1 or columns < 1:
        raise ValueError("state dimensions V and K must be positive")
    output = torch.empty(state.shape, device=state.device, dtype=torch.int8)
    scales = torch.empty(
        (slots, heads, columns), device=state.device, dtype=torch.float32
    )
    if output.numel():
        with torch_device_fn.device(state.device):
            quantize_state_kernel[(triton.cdiv(columns, 16), slots * heads)](
                state,
                output,
                scales,
                reduction,
                columns,
                triton.next_power_of_2(reduction),
                16,
                num_warps=4,
            )
    return output, scales


@libentry()
@triton.jit
def gdn_decode_int8_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    state,
    scales,
    output,
    indices,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    STRIDE_QKV: tl.constexpr,
    STRIDE_A: tl.constexpr,
    STRIDE_B: tl.constexpr,
    SCALE: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_WARPS: tl.constexpr = 4,
):
    sequence_head = tl.program_id(1).to(tl.int64)
    sequence, head = sequence_head // HV, sequence_head % HV
    query_head = head // (HV // H)
    column = tl.program_id(0) * BV + tl.arange(0, BV)
    reduction = tl.arange(0, BK)
    slot = tl.load(indices + sequence).to(tl.int64)
    output_ptr = output + (sequence * HV + head) * V + column
    # Slot zero is the null block in vLLM packed decode.
    if slot <= 0:
        tl.store(output_ptr, 0.0, column < V)
        return
    group = slot * HV + head
    offsets = (group * V + column[:, None]) * K + reduction[None, :]
    mask = (column[:, None] < V) & (reduction[None, :] < K)
    scale_offsets = group * V + column
    state_values = tl.load(state + offsets, mask, 0).to(tl.float32)
    state_values *= tl.load(scales + scale_offsets, column < V, 0)[:, None]
    base = mixed_qkv + sequence * STRIDE_QKV
    query = tl.load(base + query_head * K + reduction, reduction < K, 0).to(tl.float32)
    key = tl.load(base + H * K + query_head * K + reduction, reduction < K, 0).to(
        tl.float32
    )
    value = tl.load(base + 2 * H * K + head * V + column, column < V, 0).to(tl.float32)
    if NORMALIZE:
        query *= tl.rsqrt(tl.sum(query * query, 0) + 1e-6)
        key *= tl.rsqrt(tl.sum(key * key, 0) + 1e-6)
    query *= SCALE
    gate = tl.load(a + sequence * STRIDE_A + head).to(tl.float32)
    gate += tl.load(dt_bias + head).to(tl.float32)
    softplus = tl.where(gate <= 20.0, tl.log(1.0 + tl.exp(gate)), gate)
    decay = exp(-tl.exp(tl.load(A_log + head).to(tl.float32)) * softplus)
    beta = tl.sigmoid(tl.load(b + sequence * STRIDE_B + head).to(tl.float32))
    state_values *= decay
    residual = (value - tl.sum(state_values * key[None, :], 1)) * beta
    state_values += residual[:, None] * key[None, :]
    result = tl.sum(state_values * query[None, :], 1)
    tl.store(output_ptr, result, column < V)
    store_int8_state(
        state_values, state, scales, offsets, scale_offsets, mask, column < V
    )


def gdn_w8a16_int8(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    state_int8: torch.Tensor,
    state_scale: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed one-token-per-sequence GDN with in-place INT8 state.

    State layout follows vLLM: [slots, HV, V, K]. Scales are [slots, HV, V].
    Positive indices must be distinct and within the state allocation; nonpositive
    indices are null slots. Callers retain ownership of state and scale buffers.
    """
    logger.debug("GEMS_THEAD GDN_W8A16_INT8")
    if mixed_qkv.ndim != 2 or state_int8.ndim != 4:
        raise ValueError("expected packed QKV and four-dimensional state")
    tokens, packed = mixed_qkv.shape
    slots, heads, columns, reduction = state_int8.shape
    if columns < 1 or reduction < 1 or heads < 1 or slots < 1:
        raise ValueError("state dimensions must be positive")
    query_dim = packed - heads * columns
    if query_dim <= 0 or query_dim % (2 * reduction):
        raise ValueError("packed QKV dimensions do not match the state")
    query_heads = query_dim // (2 * reduction)
    if heads % query_heads:
        raise ValueError("value heads must be divisible by query heads")
    if mixed_qkv.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("QKV must be BF16 or FP16")
    if state_int8.dtype != torch.int8 or state_scale.dtype != torch.float32:
        raise TypeError("state and scales must be INT8 and FP32")
    if state_scale.shape != (slots, heads, columns):
        raise ValueError("state scales must have shape [slots, HV, V]")
    if a.shape != (tokens, heads) or b.shape != a.shape:
        raise ValueError("a and b must have shape [tokens, HV]")
    if a.dtype != mixed_qkv.dtype or b.dtype != mixed_qkv.dtype:
        raise TypeError("gate activations must match QKV dtype")
    if A_log.shape != (heads,) or dt_bias.shape != (heads,):
        raise ValueError("A_log and dt_bias must have shape [HV]")
    if A_log.dtype != torch.float32 or dt_bias.dtype != torch.float32:
        raise TypeError("A_log and dt_bias must be FP32")
    if out.shape != (tokens, 1, heads, columns) or out.dtype != mixed_qkv.dtype:
        raise ValueError("out must match activation dtype and shape [tokens, 1, HV, V]")
    if ssm_state_indices.shape != (tokens,) or ssm_state_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("state indices must be an integer vector of length tokens")
    tensors = (
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state_int8,
        state_scale,
        out,
        ssm_state_indices,
    )
    if mixed_qkv.device.type != "cuda" or any(
        tensor.device != mixed_qkv.device for tensor in tensors
    ):
        raise ValueError("all tensors must be on the same accelerator")
    if any(tensor.requires_grad for tensor in tensors):
        raise NotImplementedError("GDN INT8 state is forward-only")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all buffers must be contiguous")
    if not tokens:
        return out, state_int8
    # Measured PPU dispatch: small tiles expose parallelism for short decode batches.
    if tokens <= 2:
        block_v, warps = 8, 2
    elif tokens < 32:
        block_v, warps = 8, 1
    else:
        block_v, warps = 32, 2
    with torch_device_fn.device(mixed_qkv.device):
        gdn_decode_int8_kernel[(triton.cdiv(columns, block_v), tokens * heads)](
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            state_int8,
            state_scale,
            out,
            ssm_state_indices,
            query_heads,
            heads,
            reduction,
            columns,
            mixed_qkv.stride(0),
            a.stride(0),
            b.stride(0),
            scale,
            use_qk_l2norm_in_kernel,
            triton.next_power_of_2(reduction),
            block_v,
            warps,
            num_warps=warps,
            num_stages=1,
        )
    return out, state_int8
