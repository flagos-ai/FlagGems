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

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.fused.FLA.triton_ops_helper import exp
from flag_gems.utils import libentry

_STATE_BLOCK_V = 32
_PERSISTENT_BLOCK_V = 32


@triton.jit
def _fp32_to_int8_rni(values):
    return tl.inline_asm_elementwise(
        "cvt.rni.sat.s8.f32 $0, $1;",
        constraints="=r,f",
        args=[values],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    ).to(tl.int8)


@triton.jit
def _quantize_state_tile_int8(values, mask_k):
    absmax_per_v = tl.max(
        tl.where(mask_k[:, None], tl.abs(values), 0.0),
        axis=0,
    )
    absmax = tl.max(absmax_per_v, axis=0)
    state_scale = tl.where(absmax > 0.0, absmax * (1.0 / 127.0), 1.0)
    inv_scale = tl.where(absmax > 0.0, 127.0 / absmax, 0.0)
    quantized = _fp32_to_int8_rni(values * inv_scale)
    return quantized, state_scale


@triton.jit
def _quantize_state_tile_int8_v_first(values, mask_k):
    absmax_per_k = tl.max(
        tl.where(mask_k[:, None], tl.abs(values), 0.0),
        axis=1,
    )
    absmax = tl.max(absmax_per_k, axis=0)
    state_scale = tl.where(absmax > 0.0, absmax * (1.0 / 127.0), 1.0)
    inv_scale = tl.where(absmax > 0.0, 127.0 / absmax, 0.0)
    quantized = _fp32_to_int8_rni(values * inv_scale)
    return quantized, state_scale


@libentry()
@triton.jit
def _quantize_gdn_state_int8_kernel(
    state,
    state_int8,
    state_scale,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_sh = tl.program_id(1)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask = mask_k[:, None] & mask_v[None, :]
    offsets = i_sh * K * V + o_k[:, None] * V + o_v[None, :]
    values = tl.load(state + offsets, mask=mask, other=0.0).to(tl.float32)
    quantized, scales = _quantize_state_tile_int8(values, mask_k)
    tl.store(state_int8 + offsets, quantized, mask=mask)
    tl.store(state_scale + i_sh * V + o_v, scales, mask=mask_v)


@libentry()
@triton.jit
def _dequantize_gdn_state_int8_kernel(
    state_int8,
    state_scale,
    state,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_sh = tl.program_id(1)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask = (o_k[:, None] < K) & (o_v[None, :] < V)
    offsets = i_sh * K * V + o_k[:, None] * V + o_v[None, :]
    scales = tl.load(state_scale + i_sh * V + o_v, mask=o_v < V, other=1.0)
    values = tl.load(state_int8 + offsets, mask=mask, other=0.0).to(tl.float32)
    values *= scales[None, :]
    tl.store(state + offsets, values.to(state.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _fused_recurrent_gated_delta_rule_w8a16_int8_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    state_int8,
    state_scale,
    cu_seqlens,
    state_indices,
    scale,
    stride_q_b: tl.constexpr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_k: tl.constexpr,
    stride_k_b: tl.constexpr,
    stride_k_t: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_k_k: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_t: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_v: tl.constexpr,
    stride_g_b: tl.constexpr,
    stride_g_t: tl.constexpr,
    stride_g_h: tl.constexpr,
    stride_beta_b: tl.constexpr,
    stride_beta_t: tl.constexpr,
    stride_beta_h: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_v: tl.constexpr,
    stride_state_s: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_state_k: tl.constexpr,
    stride_state_v: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_PACKED_INPUT: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    if USE_PACKED_INPUT:
        i_b = 0
        i_t = tl.load(cu_seqlens + i_n).to(tl.int64)
    else:
        i_b = i_n
        i_t = 0

    if USE_STATE_INDICES:
        state_index = tl.load(state_indices + i_n).to(tl.int64)
        if state_index < 0:
            return
    else:
        state_index = i_n

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_state = mask_k[:, None] & mask_v[None, :]

    state_offsets = (
        state_index * stride_state_s
        + i_hv * stride_state_h
        + o_k[:, None] * stride_state_k
        + o_v[None, :] * stride_state_v
    )
    scale_base = (state_index * HV + i_hv) * V
    scale_offsets = scale_base + o_v
    b_state_scale = tl.load(state_scale + scale_base + i_v * BV).to(tl.float32)
    b_h = tl.load(
        state_int8 + state_offsets,
        mask=mask_state,
        other=0.0,
    ).to(tl.float32)

    q_offsets = (
        i_b * stride_q_b + i_t * stride_q_t + i_h * stride_q_h + o_k * stride_q_k
    )
    k_offsets = (
        i_b * stride_k_b + i_t * stride_k_t + i_h * stride_k_h + o_k * stride_k_k
    )
    v_offsets = (
        i_b * stride_v_b + i_t * stride_v_t + i_hv * stride_v_h + o_v * stride_v_v
    )
    b_q = tl.load(q + q_offsets, mask=mask_k, other=0.0).to(tl.float32)
    b_k = tl.load(k + k_offsets, mask=mask_k, other=0.0).to(tl.float32)
    b_v = tl.load(v + v_offsets, mask=mask_v, other=0.0).to(tl.float32)

    if USE_QK_L2NORM:
        b_q *= tl.rsqrt(tl.sum(b_q * b_q, axis=0) + 1e-6) * scale
        b_k *= tl.rsqrt(tl.sum(b_k * b_k, axis=0) + 1e-6)
    else:
        b_q *= scale

    g_offset = i_b * stride_g_b + i_t * stride_g_t + i_hv * stride_g_h
    beta_offset = i_b * stride_beta_b + i_t * stride_beta_t + i_hv * stride_beta_h
    b_h *= b_state_scale * exp(tl.load(g + g_offset).to(tl.float32))
    b_v -= tl.sum(b_h * b_k[:, None], axis=0)
    b_v *= tl.load(beta + beta_offset).to(tl.float32)
    b_h += b_k[:, None] * b_v[None, :]
    b_o = tl.sum(b_h * b_q[:, None], axis=0)

    o_offsets = (
        i_b * stride_o_b + i_t * stride_o_t + i_hv * stride_o_h + o_v * stride_o_v
    )
    tl.store(o + o_offsets, b_o.to(o.dtype.element_ty), mask=mask_v)

    quantized, b_state_scale = _quantize_state_tile_int8(b_h, mask_k)
    tl.store(state_int8 + state_offsets, quantized, mask=mask_state)
    tl.store(state_scale + scale_offsets, b_state_scale, mask=mask_v)


@libentry()
@triton.jit
def _fused_recurrent_gated_delta_rule_grouped_w8a16_int8_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    state_int8,
    state_scale,
    cu_seqlens,
    state_indices,
    scale,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_k_t: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_v_t: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_g_t: tl.constexpr,
    stride_g_h: tl.constexpr,
    stride_beta_t: tl.constexpr,
    stride_beta_h: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_state_s: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_state_k: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    BK: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    STREAM_STATE_STORE: tl.constexpr,
    REDUCE_V_FIRST: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // H
    i_h = i_nh % H
    i_hv_base = i_h * (HV // H)
    i_t = tl.load(cu_seqlens + i_n).to(tl.int64)
    state_index = tl.load(state_indices + i_n).to(tl.int64)
    if state_index < 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < BK
    b_q = tl.load(q + i_t * stride_q_t + i_h * stride_q_h + o_k).to(tl.float32)
    b_k = tl.load(k + i_t * stride_k_t + i_h * stride_k_h + o_k).to(tl.float32)
    b_q *= tl.rsqrt(tl.sum(b_q * b_q, axis=0) + 1e-6) * scale
    b_k *= tl.rsqrt(tl.sum(b_k * b_k, axis=0) + 1e-6)

    for i_group in tl.range(0, HV // H, loop_unroll_factor=1):
        i_hv = i_hv_base + i_group
        g_offset = i_t * stride_g_t + i_hv * stride_g_h
        beta_offset = i_t * stride_beta_t + i_hv * stride_beta_h
        decay = exp(tl.load(g + g_offset).to(tl.float32))
        b_beta = tl.load(beta + beta_offset).to(tl.float32)

        state_offsets = (
            state_index * stride_state_s
            + i_hv * stride_state_h
            + o_k[:, None] * stride_state_k
            + o_v[None, :]
        )
        scale_base = (state_index * HV + i_hv) * V
        scale_offsets = scale_base + o_v
        b_state_scale = tl.load(state_scale + scale_base + i_v * BV).to(tl.float32)
        b_h = tl.load(state_int8 + state_offsets).to(tl.float32)
        b_v = tl.load(v + i_t * stride_v_t + i_hv * stride_v_h + o_v).to(tl.float32)

        b_h *= b_state_scale * decay
        b_v = (b_v - tl.sum(b_h * b_k[:, None], axis=0)) * b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], axis=0)

        tl.store(
            o + i_t * stride_o_t + i_hv * stride_o_h + o_v,
            b_o.to(o.dtype.element_ty),
        )
        if REDUCE_V_FIRST:
            quantized, b_state_scale = _quantize_state_tile_int8_v_first(b_h, mask_k)
        else:
            quantized, b_state_scale = _quantize_state_tile_int8(b_h, mask_k)
        if STREAM_STATE_STORE:
            tl.store(
                state_int8 + state_offsets,
                quantized,
                cache_modifier=".cs",
            )
        else:
            tl.store(state_int8 + state_offsets, quantized)
        tl.store(state_scale + scale_offsets, b_state_scale)


@libentry()
@triton.jit
def _fused_recurrent_gated_delta_rule_persistent_w8a16_int8_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    state_int8,
    state_scale,
    cu_seqlens,
    state_indices,
    scale,
    stride_q_b: tl.constexpr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_k: tl.constexpr,
    stride_k_b: tl.constexpr,
    stride_k_t: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_k_k: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_t: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_v: tl.constexpr,
    stride_g_b: tl.constexpr,
    stride_g_t: tl.constexpr,
    stride_g_h: tl.constexpr,
    stride_beta_b: tl.constexpr,
    stride_beta_t: tl.constexpr,
    stride_beta_h: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_v: tl.constexpr,
    stride_state_s: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_state_k: tl.constexpr,
    stride_state_v: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    GROUP_HV: tl.constexpr,
    USE_PACKED_INPUT: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    STREAM_STATE_STORE: tl.constexpr,
):
    i_program = tl.program_id(0)
    programs_per_sequence = HV // GROUP_HV
    i_n = i_program // programs_per_sequence
    i_hv_base = (i_program % programs_per_sequence) * GROUP_HV
    i_h = i_hv_base // (HV // H)

    if USE_PACKED_INPUT:
        i_b = 0
        i_t = tl.load(cu_seqlens + i_n).to(tl.int64)
    else:
        i_b = i_n
        i_t = 0

    if USE_STATE_INDICES:
        state_index = tl.load(state_indices + i_n).to(tl.int64)
        if state_index < 0:
            return
    else:
        state_index = i_n

    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    q_offsets = (
        i_b * stride_q_b + i_t * stride_q_t + i_h * stride_q_h + o_k * stride_q_k
    )
    k_offsets = (
        i_b * stride_k_b + i_t * stride_k_t + i_h * stride_k_h + o_k * stride_k_k
    )
    b_q = tl.load(q + q_offsets, mask=mask_k, other=0.0).to(tl.float32)
    b_k = tl.load(k + k_offsets, mask=mask_k, other=0.0).to(tl.float32)
    if USE_QK_L2NORM:
        b_q *= tl.rsqrt(tl.sum(b_q * b_q, axis=0) + 1e-6) * scale
        b_k *= tl.rsqrt(tl.sum(b_k * b_k, axis=0) + 1e-6)
    else:
        b_q *= scale

    for i_group in tl.range(0, GROUP_HV, loop_unroll_factor=1):
        i_hv = i_hv_base + i_group
        g_offset = i_b * stride_g_b + i_t * stride_g_t + i_hv * stride_g_h
        beta_offset = i_b * stride_beta_b + i_t * stride_beta_t + i_hv * stride_beta_h
        decay = exp(tl.load(g + g_offset).to(tl.float32))
        b_beta = tl.load(beta + beta_offset).to(tl.float32)

        for i_v in tl.range(0, NV, loop_unroll_factor=1):
            o_v = i_v * BV + tl.arange(0, BV)
            mask_v = o_v < V
            mask_state = mask_k[:, None] & mask_v[None, :]
            state_offsets = (
                state_index * stride_state_s
                + i_hv * stride_state_h
                + o_k[:, None] * stride_state_k
                + o_v[None, :] * stride_state_v
            )
            scale_base = (state_index * HV + i_hv) * V
            scale_offsets = scale_base + o_v
            b_state_scale = tl.load(state_scale + scale_base + i_v * BV).to(tl.float32)
            b_h = tl.load(
                state_int8 + state_offsets,
                mask=mask_state,
                other=0.0,
            ).to(tl.float32)
            v_offsets = (
                i_b * stride_v_b
                + i_t * stride_v_t
                + i_hv * stride_v_h
                + o_v * stride_v_v
            )
            b_v = tl.load(v + v_offsets, mask=mask_v, other=0.0).to(tl.float32)

            b_h *= b_state_scale * decay
            b_v = (b_v - tl.sum(b_h * b_k[:, None], axis=0)) * b_beta
            b_h += b_k[:, None] * b_v[None, :]
            b_o = tl.sum(b_h * b_q[:, None], axis=0)

            o_offsets = (
                i_b * stride_o_b
                + i_t * stride_o_t
                + i_hv * stride_o_h
                + o_v * stride_o_v
            )
            tl.store(o + o_offsets, b_o.to(o.dtype.element_ty), mask=mask_v)

            state_values, b_state_scale = _quantize_state_tile_int8(b_h, mask_k)
            if STREAM_STATE_STORE:
                tl.store(
                    state_int8 + state_offsets,
                    state_values,
                    mask=mask_state,
                    cache_modifier=".cs",
                )
            else:
                tl.store(
                    state_int8 + state_offsets,
                    state_values,
                    mask=mask_state,
                )
            tl.store(state_scale + scale_offsets, b_state_scale, mask=mask_v)


def quantize_gdn_state_int8(
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a contiguous [S, HV, K, V] GDN state per V tile."""
    if state.ndim != 4 or not state.is_contiguous():
        raise ValueError("state must be a contiguous [S, HV, K, V] tensor")

    S, HV, K, V = state.shape
    state_int8 = torch.empty_like(state, dtype=torch.int8)
    state_scale = torch.empty((S, HV, V), device=state.device, dtype=torch.float32)
    _quantize_gdn_state_int8_kernel[(triton.cdiv(V, _STATE_BLOCK_V), S * HV)](
        state,
        state_int8,
        state_scale,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=_STATE_BLOCK_V,
        num_warps=4,
        num_stages=1,
    )
    return state_int8, state_scale


def dequantize_gdn_state_int8(
    state_int8: torch.Tensor,
    state_scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize an INT8 GDN state using replicated per-tile scales."""
    if state_int8.ndim != 4 or not state_int8.is_contiguous():
        raise ValueError("state_int8 must be a contiguous [S, HV, K, V] tensor")
    S, HV, K, V = state_int8.shape
    if (
        state_int8.dtype != torch.int8
        or state_scale.dtype != torch.float32
        or not state_scale.is_contiguous()
        or state_scale.shape != (S, HV, V)
        or state_scale.device != state_int8.device
    ):
        raise ValueError("state_scale must be contiguous FP32 with shape [S, HV, V]")
    state = torch.empty(state_int8.shape, device=state_int8.device, dtype=output_dtype)
    _dequantize_gdn_state_int8_kernel[
        (
            triton.cdiv(V, _STATE_BLOCK_V),
            S * HV,
        )
    ](
        state_int8,
        state_scale,
        state,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=_STATE_BLOCK_V,
        num_warps=4,
        num_stages=1,
    )
    return state


def fused_recurrent_gated_delta_rule_w8a16_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_int8: torch.Tensor,
    state_scale: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run single-token GDN decode with INT8 state and BF16/FP16 activations.

    Packed inputs use shape [1, N, H, K] and one token per sequence in
    ``cu_seqlens``. Non-packed inputs use shape [N, 1, H, K]. The recurrent
    state and its replicated per-tile FP32 scales are updated in-place while
    all arithmetic and accumulation are performed in FP32.
    """
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("q/k/v must be 4D and q/k must have matching shapes")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q/k/v must use float16 or bfloat16 activations")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v must use the same dtype")

    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if v.shape[:2] != (B, T) or HV % H != 0:
        raise ValueError("v must share B/T with q and HV must be divisible by H")

    use_packed_input = cu_seqlens is not None
    N = B if cu_seqlens is None else cu_seqlens.numel() - 1
    if use_packed_input:
        if B != 1 or T != N:
            raise ValueError("packed decode requires shape [1, N, ...]")
    elif T != 1:
        raise ValueError("non-packed decode requires shape [N, 1, ...]")

    if state_int8.dtype != torch.int8:
        raise ValueError("state_int8 must have dtype torch.int8")
    if not state_int8.is_contiguous() or state_int8.shape[1:] != (HV, K, V):
        raise ValueError("state_int8 must be contiguous with shape [S, HV, K, V]")
    if (
        state_scale.dtype != torch.float32
        or not state_scale.is_contiguous()
        or state_scale.shape != (state_int8.shape[0], HV, V)
        or state_scale.device != state_int8.device
    ):
        raise ValueError("state_scale must be contiguous FP32 with shape [S, HV, V]")
    if state_indices is not None and state_indices.numel() != N:
        raise ValueError("state_indices must contain one index per sequence")

    o = torch.empty(B, T, HV, V, device=q.device, dtype=q.dtype)
    BK = triton.next_power_of_2(K)
    kernel_args = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "o": o,
        "state_int8": state_int8,
        "state_scale": state_scale,
        "cu_seqlens": cu_seqlens,
        "state_indices": state_indices,
        "scale": scale,
        "stride_q_b": q.stride(0),
        "stride_q_t": q.stride(1),
        "stride_q_h": q.stride(2),
        "stride_q_k": q.stride(3),
        "stride_k_b": k.stride(0),
        "stride_k_t": k.stride(1),
        "stride_k_h": k.stride(2),
        "stride_k_k": k.stride(3),
        "stride_v_b": v.stride(0),
        "stride_v_t": v.stride(1),
        "stride_v_h": v.stride(2),
        "stride_v_v": v.stride(3),
        "stride_g_b": g.stride(0),
        "stride_g_t": g.stride(1),
        "stride_g_h": g.stride(2),
        "stride_beta_b": beta.stride(0),
        "stride_beta_t": beta.stride(1),
        "stride_beta_h": beta.stride(2),
        "stride_o_b": o.stride(0),
        "stride_o_t": o.stride(1),
        "stride_o_h": o.stride(2),
        "stride_o_v": o.stride(3),
        "stride_state_s": state_int8.stride(0),
        "stride_state_h": state_int8.stride(1),
        "stride_state_k": state_int8.stride(2),
        "stride_state_v": state_int8.stride(3),
        "H": H,
        "HV": HV,
        "K": K,
        "V": V,
        "BK": BK,
        "USE_PACKED_INPUT": use_packed_input,
        "USE_STATE_INDICES": state_indices is not None,
        "USE_QK_L2NORM": use_qk_l2norm_in_kernel,
    }

    optimized_shape = K == 128 and V == 128
    use_grouped_kernel = (
        optimized_shape
        and (N < 80 or N >= 96)
        and use_packed_input
        and state_indices is not None
        and use_qk_l2norm_in_kernel
        and HV // H == 2
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
        and g.stride(-1) == 1
        and beta.stride(-1) == 1
    )
    # Group value heads so each CTA can reuse normalized q/k.
    if use_grouped_kernel:
        block_v = _STATE_BLOCK_V
        if N == 1:
            grouped_maxnreg = 255
        elif N >= 128:
            grouped_maxnreg = 160
        else:
            grouped_maxnreg = 192
        _fused_recurrent_gated_delta_rule_grouped_w8a16_int8_kernel[
            (triton.cdiv(V, block_v), N * H)
        ](
            q,
            k,
            v,
            g,
            beta,
            o,
            state_int8,
            state_scale,
            cu_seqlens,
            state_indices,
            scale,
            q.stride(1),
            q.stride(2),
            k.stride(1),
            k.stride(2),
            v.stride(1),
            v.stride(2),
            g.stride(1),
            g.stride(2),
            beta.stride(1),
            beta.stride(2),
            o.stride(1),
            o.stride(2),
            state_int8.stride(0),
            state_int8.stride(1),
            state_int8.stride(2),
            H,
            HV,
            BK,
            V,
            block_v,
            STREAM_STATE_STORE=N >= 512,
            REDUCE_V_FIRST=N >= 128,
            num_warps=1,
            num_stages=1,
            maxnreg=grouped_maxnreg,
        )
    # The persistent fallback reuses normalized q/k across V tiles.
    elif optimized_shape and (48 <= N < 80 or N >= 96):
        group_hv = 2 if 96 <= N < 160 and HV // H == 2 else 1
        _fused_recurrent_gated_delta_rule_persistent_w8a16_int8_kernel[
            (N * (HV // group_hv),)
        ](
            **kernel_args,
            BV=_PERSISTENT_BLOCK_V,
            NV=triton.cdiv(V, _PERSISTENT_BLOCK_V),
            GROUP_HV=group_hv,
            STREAM_STATE_STORE=N >= 512,
            num_warps=1,
            num_stages=1,
        )
    else:
        block_v = _STATE_BLOCK_V
        if K == 128 and V == 128:
            if N == 1:
                block_v = 8
            elif N == 2:
                block_v = 16
        NV = triton.cdiv(V, block_v)
        _fused_recurrent_gated_delta_rule_w8a16_int8_kernel[(NV, N * HV)](
            **kernel_args,
            BV=block_v,
            num_warps=1,
            num_stages=1,
        )
    return o, state_int8, state_scale
