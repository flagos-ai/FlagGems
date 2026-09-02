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

from flag_gems.fused.FLA.chunk import chunk_gated_delta_rule_fwd
from flag_gems.fused.FLA.triton_ops_helper import exp
from flag_gems.fused.FLA.utils import is_tma_supported
from flag_gems.utils import libentry

_STATE_BLOCK_V = 32
_FP8_E4M3_MAX = tl.constexpr(448.0)


@triton.jit
def _load_dynamic_fp8_state(
    state_fp8,
    state_scale,
    state_offsets,
    scale_offsets,
    mask_state,
    mask_v,
    ACTIVATION_TYPE: tl.constexpr,
):
    values = tl.load(state_fp8 + state_offsets, mask=mask_state, other=0.0).to(
        ACTIVATION_TYPE
    )
    channel_scale = tl.load(state_scale + scale_offsets, mask=mask_v, other=0.0).to(
        ACTIVATION_TYPE
    )
    return (values * channel_scale[None, :]).to(tl.float32)


@triton.jit
def _store_dynamic_fp8_state(
    values,
    state_fp8,
    state_scale,
    state_offsets,
    scale_offsets,
    mask_state,
    mask_v,
    ACTIVATION_TYPE: tl.constexpr,
    STREAM_STATE_STORE: tl.constexpr,
    LOW_PRECISION_AMAX: tl.constexpr,
):
    # One scale per V channel. The reciprocal is scalar per channel, avoiding a
    # division for every state element while retaining the full tile range.
    if LOW_PRECISION_AMAX:
        # Reduce the values that will actually be cast to FP8. This also shortens
        # the FP32 state tile's live range in large decode grids.
        values_low = values.to(ACTIVATION_TYPE)
        channel_amax = tl.max(tl.abs(values_low), axis=0).to(tl.float32)
    else:
        channel_amax = tl.max(tl.abs(values), axis=0)
        values_low = values.to(ACTIVATION_TYPE)
    safe_amax = tl.where(channel_amax > 0.0, channel_amax, _FP8_E4M3_MAX)
    channel_scale = safe_amax / _FP8_E4M3_MAX
    inv_scale = _FP8_E4M3_MAX / safe_amax
    quantized = (values_low * inv_scale.to(ACTIVATION_TYPE)[None, :]).to(
        state_fp8.dtype.element_ty
    )

    tl.store(state_scale + scale_offsets, channel_scale, mask=mask_v)
    if STREAM_STATE_STORE:
        tl.store(
            state_fp8 + state_offsets,
            quantized,
            mask=mask_state,
            cache_modifier=".cs",
        )
    else:
        tl.store(state_fp8 + state_offsets, quantized, mask=mask_state)


@triton.jit
def _prepare_qk(
    q_values,
    k_values,
    scale,
    ACTIVATION_TYPE: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
):
    q_low = q_values.to(ACTIVATION_TYPE)
    k_low = k_values.to(ACTIVATION_TYPE)
    if USE_QK_L2NORM:
        q_square = (q_low * q_low).to(tl.float32)
        k_square = (k_low * k_low).to(tl.float32)
        q_norm = tl.rsqrt(tl.sum(q_square, axis=0) + 1.0e-6)
        k_norm = tl.rsqrt(tl.sum(k_square, axis=0) + 1.0e-6)
        q_factor = (q_norm.to(ACTIVATION_TYPE) * scale.to(ACTIVATION_TYPE)).to(
            ACTIVATION_TYPE
        )
        q_low = (q_low * q_factor).to(ACTIVATION_TYPE)
        k_low = (k_low * k_norm.to(ACTIVATION_TYPE)).to(ACTIVATION_TYPE)
    else:
        q_low = (q_low * scale.to(ACTIVATION_TYPE)).to(ACTIVATION_TYPE)
    return q_low, k_low


@libentry()
@triton.jit
def _normalize_prefill_qk_kernel(
    q,
    k,
    q_normalized,
    k_normalized,
    unit_scale,
    stride_q_b: tl.constexpr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_k: tl.constexpr,
    stride_k_b: tl.constexpr,
    stride_k_t: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_k_k: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    i_bth = tl.program_id(0)
    i_bt = i_bth // H
    i_h = i_bth % H
    i_b = i_bt // T
    i_t = i_bt % T
    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    q_offsets = (
        i_b * stride_q_b + i_t * stride_q_t + i_h * stride_q_h + o_k * stride_q_k
    )
    k_offsets = (
        i_b * stride_k_b + i_t * stride_k_t + i_h * stride_k_h + o_k * stride_k_k
    )
    q_values = tl.load(q + q_offsets, mask=mask_k, other=0.0)
    k_values = tl.load(k + k_offsets, mask=mask_k, other=0.0)
    q_values, k_values = _prepare_qk(
        q_values,
        k_values,
        unit_scale,
        ACTIVATION_TYPE=q.dtype.element_ty,
        USE_QK_L2NORM=True,
    )
    output_offsets = i_bth * K + o_k
    tl.store(q_normalized + output_offsets, q_values, mask=mask_k)
    tl.store(k_normalized + output_offsets, k_values, mask=mask_k)


@triton.jit
def _gdn_step(
    state_acc,
    q_low,
    k_low,
    v_values,
    decay,
    beta_value,
    ACTIVATION_TYPE: tl.constexpr,
    FP32_STATE_PRODUCTS: tl.constexpr,
    FP32_UPDATE_PRODUCTS: tl.constexpr,
):
    if FP32_STATE_PRODUCTS:
        state_acc *= decay.to(tl.float32)
        prediction_product = state_acc * k_low.to(tl.float32)[:, None]
    else:
        state_acc = (state_acc.to(ACTIVATION_TYPE) * decay.to(ACTIVATION_TYPE)).to(
            tl.float32
        )
        prediction_product = (state_acc.to(ACTIVATION_TYPE) * k_low[:, None]).to(
            tl.float32
        )
    residual = v_values.to(tl.float32) - tl.sum(prediction_product, axis=0)
    if FP32_UPDATE_PRODUCTS:
        residual *= beta_value.to(tl.float32)
        update_product = k_low.to(tl.float32)[:, None] * residual[None, :]
    else:
        residual = (residual.to(ACTIVATION_TYPE) * beta_value.to(ACTIVATION_TYPE)).to(
            tl.float32
        )
        update_product = (k_low[:, None] * residual.to(ACTIVATION_TYPE)[None, :]).to(
            tl.float32
        )
    state_acc += update_product
    if FP32_STATE_PRODUCTS:
        output_product = state_acc * q_low.to(tl.float32)[:, None]
    else:
        output_product = (state_acc.to(ACTIVATION_TYPE) * q_low[:, None]).to(tl.float32)
    output = tl.sum(output_product, axis=0)
    return state_acc, output


@libentry()
@triton.jit
def _quantize_gdn_state_fp8_kernel(
    state,
    state_fp8,
    state_scale,
    K: tl.constexpr,
    V: tl.constexpr,
    HV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_3D_GRID: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    i_v = tl.program_id(0)
    if USE_3D_GRID:
        i_sh = tl.program_id(2) * HV + tl.program_id(1)
    else:
        i_sh = tl.program_id(1)
    if USE_INT64_OFFSETS:
        i_sh = i_sh.to(tl.int64)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask = (o_k[:, None] < K) & (o_v[None, :] < V)
    offsets = i_sh * K * V + o_k[:, None] * V + o_v[None, :]
    values = tl.load(state + offsets, mask=mask, other=0.0).to(tl.float32)
    scale_offsets = i_sh * V + o_v
    _store_dynamic_fp8_state(
        values,
        state_fp8,
        state_scale,
        offsets,
        scale_offsets,
        mask,
        o_v < V,
        ACTIVATION_TYPE=state.dtype.element_ty,
        STREAM_STATE_STORE=False,
        LOW_PRECISION_AMAX=False,
    )


@libentry()
@triton.jit
def _dequantize_gdn_state_fp8_kernel(
    state_fp8,
    state_scale,
    state,
    K: tl.constexpr,
    V: tl.constexpr,
    HV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_3D_GRID: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    i_v = tl.program_id(0)
    if USE_3D_GRID:
        i_sh = tl.program_id(2) * HV + tl.program_id(1)
    else:
        i_sh = tl.program_id(1)
    if USE_INT64_OFFSETS:
        i_sh = i_sh.to(tl.int64)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask = (o_k[:, None] < K) & (o_v[None, :] < V)
    offsets = i_sh * K * V + o_k[:, None] * V + o_v[None, :]
    scale_offsets = i_sh * V + o_v
    values = _load_dynamic_fp8_state(
        state_fp8,
        state_scale,
        offsets,
        scale_offsets,
        mask,
        o_v < V,
        ACTIVATION_TYPE=state.dtype.element_ty,
    )
    tl.store(state + offsets, values.to(state.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _dequantize_indexed_gdn_state_fp8_kernel(
    state_fp8,
    state_scale,
    state,
    state_indices,
    stride_state_s: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_state_k: tl.constexpr,
    stride_state_v: tl.constexpr,
    stride_scale_s: tl.constexpr,
    stride_scale_h: tl.constexpr,
    stride_scale_v: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    if USE_STATE_INDICES:
        state_index = tl.load(state_indices + i_n).to(tl.int64)
    else:
        state_index = i_n
    valid_state = state_index >= 0
    state_index = tl.where(valid_state, state_index, 0)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_state = mask_k[:, None] & mask_v[None, :] & valid_state
    source_offsets = (
        state_index * stride_state_s
        + i_hv * stride_state_h
        + o_k[:, None] * stride_state_k
        + o_v[None, :] * stride_state_v
    )
    scale_offsets = (
        state_index * stride_scale_s + i_hv * stride_scale_h + o_v * stride_scale_v
    )
    values = _load_dynamic_fp8_state(
        state_fp8,
        state_scale,
        source_offsets,
        scale_offsets,
        mask_state,
        mask_v & valid_state,
        ACTIVATION_TYPE=state.dtype.element_ty,
    )
    output_offsets = i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    tl.store(
        state + output_offsets,
        values.to(state.dtype.element_ty),
        mask=mask_k[:, None] & mask_v[None, :],
    )


@libentry()
@triton.jit
def _quantize_indexed_gdn_state_fp8_kernel(
    state,
    state_fp8,
    state_scale,
    state_indices,
    stride_state_s: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_state_k: tl.constexpr,
    stride_state_v: tl.constexpr,
    stride_scale_s: tl.constexpr,
    stride_scale_h: tl.constexpr,
    stride_scale_v: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
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
    source_offsets = i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    output_offsets = (
        state_index * stride_state_s
        + i_hv * stride_state_h
        + o_k[:, None] * stride_state_k
        + o_v[None, :] * stride_state_v
    )
    scale_offsets = (
        state_index * stride_scale_s + i_hv * stride_scale_h + o_v * stride_scale_v
    )
    values = tl.load(state + source_offsets, mask=mask_state, other=0.0).to(tl.float32)
    _store_dynamic_fp8_state(
        values,
        state_fp8,
        state_scale,
        output_offsets,
        scale_offsets,
        mask_state,
        mask_v,
        ACTIVATION_TYPE=state.dtype.element_ty,
        STREAM_STATE_STORE=False,
        LOW_PRECISION_AMAX=False,
    )


@triton.jit
def _gdn_decode_value_tile(
    v,
    o,
    state_fp8,
    state_scale,
    q_low,
    k_low,
    decay,
    beta_value,
    i_b,
    i_t,
    i_hv,
    i_v,
    initial_state_index,
    output_state_index,
    output_state_valid,
    stride_v_b: tl.constexpr,
    stride_v_t: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_v: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_v: tl.constexpr,
    stride_state_s: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_state_k: tl.constexpr,
    stride_state_v: tl.constexpr,
    stride_scale_s: tl.constexpr,
    stride_scale_h: tl.constexpr,
    stride_scale_v: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    LOW_PRECISION_AMAX: tl.constexpr,
    FP32_STATE_PRODUCTS: tl.constexpr,
    FP32_UPDATE_PRODUCTS: tl.constexpr,
):
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_state = mask_k[:, None] & mask_v[None, :]

    initial_state_offsets = (
        initial_state_index * stride_state_s
        + i_hv * stride_state_h
        + o_k[:, None] * stride_state_k
        + o_v[None, :] * stride_state_v
    )
    initial_scale_offsets = (
        initial_state_index * stride_scale_s
        + i_hv * stride_scale_h
        + o_v * stride_scale_v
    )
    state_acc = _load_dynamic_fp8_state(
        state_fp8,
        state_scale,
        initial_state_offsets,
        initial_scale_offsets,
        mask_state,
        mask_v,
        ACTIVATION_TYPE=v.dtype.element_ty,
    )
    v_offsets = (
        i_b * stride_v_b + i_t * stride_v_t + i_hv * stride_v_h + o_v * stride_v_v
    )
    v_values = tl.load(v + v_offsets, mask=mask_v, other=0.0)
    state_acc, output = _gdn_step(
        state_acc,
        q_low,
        k_low,
        v_values,
        decay,
        beta_value,
        ACTIVATION_TYPE=v.dtype.element_ty,
        FP32_STATE_PRODUCTS=FP32_STATE_PRODUCTS,
        FP32_UPDATE_PRODUCTS=FP32_UPDATE_PRODUCTS,
    )

    output_offsets = (
        i_b * stride_o_b + i_t * stride_o_t + i_hv * stride_o_h + o_v * stride_o_v
    )
    tl.store(o + output_offsets, output.to(o.dtype.element_ty), mask=mask_v)
    safe_output_state_index = tl.where(output_state_valid, output_state_index, 0)
    output_state_offsets = (
        safe_output_state_index * stride_state_s
        + i_hv * stride_state_h
        + o_k[:, None] * stride_state_k
        + o_v[None, :] * stride_state_v
    )
    output_scale_offsets = (
        safe_output_state_index * stride_scale_s
        + i_hv * stride_scale_h
        + o_v * stride_scale_v
    )
    _store_dynamic_fp8_state(
        state_acc,
        state_fp8,
        state_scale,
        output_state_offsets,
        output_scale_offsets,
        mask_state & output_state_valid,
        mask_v & output_state_valid,
        ACTIVATION_TYPE=v.dtype.element_ty,
        STREAM_STATE_STORE=False,
        LOW_PRECISION_AMAX=LOW_PRECISION_AMAX,
    )


@libentry()
@triton.jit
def _fused_recurrent_gated_delta_rule_grouped_w8a16_fp8_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    state_fp8,
    state_scale,
    cu_seqlens,
    state_indices,
    num_accepted_tokens,
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
    stride_scale_s: tl.constexpr,
    stride_scale_h: tl.constexpr,
    stride_scale_v: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_PACKED_INPUT: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
    USE_SPEC_DECODING: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    LOW_PRECISION_AMAX: tl.constexpr,
    FP32_STATE_PRODUCTS: tl.constexpr,
    FP32_UPDATE_PRODUCTS: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // H
    i_h = i_nh % H

    if USE_PACKED_INPUT:
        i_b = 0
        if N >= 128:
            i_t = tl.load(cu_seqlens + i_n).to(tl.int32)
        else:
            i_t = tl.load(cu_seqlens + i_n).to(tl.int64)
    else:
        i_b = i_n
        i_t = 0

    if USE_STATE_INDICES:
        output_index_offset = i_n * stride_indices_seq
        if N >= 128:
            output_state_index = tl.load(state_indices + output_index_offset).to(
                tl.int32
            )
            if USE_SPEC_DECODING:
                initial_token_offset = (
                    tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
                )
                initial_index_offset = (
                    i_n * stride_indices_seq + initial_token_offset * stride_indices_tok
                )
                initial_state_index = tl.load(state_indices + initial_index_offset).to(
                    tl.int32
                )
            else:
                initial_state_index = output_state_index
        else:
            output_state_index = tl.load(state_indices + output_index_offset).to(
                tl.int64
            )
            if USE_SPEC_DECODING:
                initial_token_offset = (
                    tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
                )
                initial_index_offset = (
                    i_n * stride_indices_seq + initial_token_offset * stride_indices_tok
                )
                initial_state_index = tl.load(state_indices + initial_index_offset).to(
                    tl.int64
                )
            else:
                initial_state_index = output_state_index
        if initial_state_index < 0:
            return
        output_state_valid = output_state_index >= 0
    else:
        initial_state_index = i_n
        output_state_index = i_n
        output_state_valid = True

    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    q_offsets = (
        i_b * stride_q_b + i_t * stride_q_t + i_h * stride_q_h + o_k * stride_q_k
    )
    k_offsets = (
        i_b * stride_k_b + i_t * stride_k_t + i_h * stride_k_h + o_k * stride_k_k
    )
    q_low, k_low = _prepare_qk(
        tl.load(q + q_offsets, mask=mask_k, other=0.0),
        tl.load(k + k_offsets, mask=mask_k, other=0.0),
        scale,
        ACTIVATION_TYPE=q.dtype.element_ty,
        USE_QK_L2NORM=USE_QK_L2NORM,
    )

    GROUP_SIZE: tl.constexpr = HV // H
    for group_offset in tl.static_range(0, GROUP_SIZE):
        i_hv = i_h * GROUP_SIZE + group_offset
        g_offset = i_b * stride_g_b + i_t * stride_g_t + i_hv * stride_g_h
        beta_offset = i_b * stride_beta_b + i_t * stride_beta_t + i_hv * stride_beta_h
        _gdn_decode_value_tile(
            v,
            o,
            state_fp8,
            state_scale,
            q_low,
            k_low,
            exp(tl.load(g + g_offset).to(tl.float32)),
            tl.load(beta + beta_offset),
            i_b,
            i_t,
            i_hv,
            i_v,
            initial_state_index,
            output_state_index,
            output_state_valid,
            stride_v_b,
            stride_v_t,
            stride_v_h,
            stride_v_v,
            stride_o_b,
            stride_o_t,
            stride_o_h,
            stride_o_v,
            stride_state_s,
            stride_state_h,
            stride_state_k,
            stride_state_v,
            stride_scale_s,
            stride_scale_h,
            stride_scale_v,
            K,
            V,
            BK,
            BV,
            LOW_PRECISION_AMAX,
            FP32_STATE_PRODUCTS,
            FP32_UPDATE_PRODUCTS,
        )


@libentry()
@triton.jit
def _fused_recurrent_gated_delta_rule_w8a16_fp8_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    state_fp8,
    state_scale,
    cu_seqlens,
    state_indices,
    num_accepted_tokens,
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
    stride_scale_s: tl.constexpr,
    stride_scale_h: tl.constexpr,
    stride_scale_v: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_PACKED_INPUT: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
    USE_SPEC_DECODING: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    USE_3D_GRID: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_INT64_STATE_OFFSETS: tl.constexpr,
    LOW_PRECISION_AMAX: tl.constexpr,
    FP32_STATE_PRODUCTS: tl.constexpr,
    FP32_UPDATE_PRODUCTS: tl.constexpr,
):
    i_v = tl.program_id(0)
    if USE_3D_GRID:
        i_hv = tl.program_id(1)
        i_n = tl.program_id(2)
        i_nh = i_n * HV + i_hv
    else:
        i_nh = tl.program_id(1)
        i_n = i_nh // HV
        i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    if USE_PACKED_INPUT:
        i_b = 0
        if N >= 128:
            i_t = tl.load(cu_seqlens + i_n).to(tl.int32)
        else:
            i_t = tl.load(cu_seqlens + i_n).to(tl.int64)
    else:
        i_b = i_n
        i_t = 0

    if USE_STATE_INDICES:
        output_index_offset = i_n * stride_indices_seq
        if N >= 128:
            output_state_index = tl.load(state_indices + output_index_offset).to(
                tl.int32
            )
            if USE_SPEC_DECODING:
                initial_token_offset = (
                    tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
                )
                initial_index_offset = (
                    i_n * stride_indices_seq + initial_token_offset * stride_indices_tok
                )
                initial_state_index = tl.load(state_indices + initial_index_offset).to(
                    tl.int32
                )
            else:
                initial_state_index = output_state_index
        else:
            output_state_index = tl.load(state_indices + output_index_offset).to(
                tl.int64
            )
            if USE_SPEC_DECODING:
                initial_token_offset = (
                    tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
                )
                initial_index_offset = (
                    i_n * stride_indices_seq + initial_token_offset * stride_indices_tok
                )
                initial_state_index = tl.load(state_indices + initial_index_offset).to(
                    tl.int64
                )
            else:
                initial_state_index = output_state_index
        if initial_state_index < 0:
            return
        output_state_valid = output_state_index >= 0
    else:
        initial_state_index = i_n
        output_state_index = i_n
        output_state_valid = True

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_state = mask_k[:, None] & mask_v[None, :]

    if USE_INT64_STATE_OFFSETS:
        initial_state_index_for_offset = initial_state_index.to(tl.int64)
    else:
        initial_state_index_for_offset = initial_state_index
    initial_state_offsets = (
        initial_state_index_for_offset * stride_state_s
        + i_hv * stride_state_h
        + o_k[:, None] * stride_state_k
        + o_v[None, :] * stride_state_v
    )
    initial_scale_offsets = (
        initial_state_index * stride_scale_s
        + i_hv * stride_scale_h
        + o_v * stride_scale_v
    )
    safe_output_state_index = tl.where(output_state_valid, output_state_index, 0)
    if USE_INT64_STATE_OFFSETS:
        output_state_index_for_offset = safe_output_state_index.to(tl.int64)
    else:
        output_state_index_for_offset = safe_output_state_index
    output_state_offsets = (
        output_state_index_for_offset * stride_state_s
        + i_hv * stride_state_h
        + o_k[:, None] * stride_state_k
        + o_v[None, :] * stride_state_v
    )
    output_scale_offsets = (
        safe_output_state_index * stride_scale_s
        + i_hv * stride_scale_h
        + o_v * stride_scale_v
    )
    if USE_TMA_LOAD:
        state_descriptor = tl.make_tensor_descriptor(
            state_fp8,
            shape=[S * HV, K, V],
            strides=[stride_state_h, stride_state_k, stride_state_v],
            block_shape=[1, BK, BV],
        )
        state_row = (initial_state_index * HV + i_hv).to(tl.int32)
        state_values = state_descriptor.load([state_row, 0, i_v * BV])
        state_values = tl.reshape(state_values, (BK, BV))
        channel_scale = tl.load(
            state_scale + initial_scale_offsets, mask=mask_v, other=0.0
        ).to(q.dtype.element_ty)
        b_h = (state_values.to(q.dtype.element_ty) * channel_scale[None, :]).to(
            tl.float32
        )
    else:
        b_h = _load_dynamic_fp8_state(
            state_fp8,
            state_scale,
            initial_state_offsets,
            initial_scale_offsets,
            mask_state,
            mask_v,
            ACTIVATION_TYPE=q.dtype.element_ty,
        )

    q_offsets = (
        i_b * stride_q_b + i_t * stride_q_t + i_h * stride_q_h + o_k * stride_q_k
    )
    k_offsets = (
        i_b * stride_k_b + i_t * stride_k_t + i_h * stride_k_h + o_k * stride_k_k
    )
    v_offsets = (
        i_b * stride_v_b + i_t * stride_v_t + i_hv * stride_v_h + o_v * stride_v_v
    )
    b_q = tl.load(q + q_offsets, mask=mask_k, other=0.0)
    b_k = tl.load(k + k_offsets, mask=mask_k, other=0.0)
    b_v = tl.load(v + v_offsets, mask=mask_v, other=0.0)

    b_q, b_k = _prepare_qk(
        b_q,
        b_k,
        scale,
        ACTIVATION_TYPE=q.dtype.element_ty,
        USE_QK_L2NORM=USE_QK_L2NORM,
    )

    g_offset = i_b * stride_g_b + i_t * stride_g_t + i_hv * stride_g_h
    beta_offset = i_b * stride_beta_b + i_t * stride_beta_t + i_hv * stride_beta_h
    b_h, b_o = _gdn_step(
        b_h,
        b_q,
        b_k,
        b_v,
        exp(tl.load(g + g_offset).to(tl.float32)),
        tl.load(beta + beta_offset),
        ACTIVATION_TYPE=q.dtype.element_ty,
        FP32_STATE_PRODUCTS=FP32_STATE_PRODUCTS,
        FP32_UPDATE_PRODUCTS=FP32_UPDATE_PRODUCTS,
    )

    o_offsets = (
        i_b * stride_o_b + i_t * stride_o_t + i_hv * stride_o_h + o_v * stride_o_v
    )
    tl.store(o + o_offsets, b_o.to(o.dtype.element_ty), mask=mask_v)

    _store_dynamic_fp8_state(
        b_h,
        state_fp8,
        state_scale,
        output_state_offsets,
        output_scale_offsets,
        mask_state & output_state_valid,
        mask_v & output_state_valid,
        ACTIVATION_TYPE=q.dtype.element_ty,
        STREAM_STATE_STORE=False,
        LOW_PRECISION_AMAX=LOW_PRECISION_AMAX,
    )


@libentry()
@triton.jit(do_not_specialize=["T"])
def _fused_recurrent_gated_delta_rule_sequence_w8a16_fp8_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    state_fp8,
    state_scale,
    cu_seqlens,
    state_indices,
    num_accepted_tokens,
    scale,
    T: tl.int64,
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
    stride_scale_s: tl.constexpr,
    stride_scale_h: tl.constexpr,
    stride_scale_v: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_PACKED_INPUT: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
    USE_PER_TOKEN_STATE_INDICES: tl.constexpr,
    USE_SPEC_DECODING: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    LOW_PRECISION_AMAX: tl.constexpr,
    FP32_STATE_PRODUCTS: tl.constexpr,
    FP32_UPDATE_PRODUCTS: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    if USE_PACKED_INPUT:
        i_b = 0
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        sequence_length = eos - bos
    else:
        i_b = i_n
        bos = 0
        sequence_length = T

    if sequence_length <= 0:
        return

    if USE_STATE_INDICES:
        initial_token_offset = 0
        if USE_SPEC_DECODING:
            initial_token_offset = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
        initial_state_index = tl.load(
            state_indices
            + i_n * stride_indices_seq
            + initial_token_offset * stride_indices_tok
        ).to(tl.int64)
        if initial_state_index < 0:
            return
    else:
        initial_state_index = i_n

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_state = mask_k[:, None] & mask_v[None, :]
    initial_state_offsets = (
        initial_state_index * stride_state_s
        + i_hv * stride_state_h
        + o_k[:, None] * stride_state_k
        + o_v[None, :] * stride_state_v
    )
    initial_scale_offsets = (
        initial_state_index * stride_scale_s
        + i_hv * stride_scale_h
        + o_v * stride_scale_v
    )
    state_acc = _load_dynamic_fp8_state(
        state_fp8,
        state_scale,
        initial_state_offsets,
        initial_scale_offsets,
        mask_state,
        mask_v,
        ACTIVATION_TYPE=q.dtype.element_ty,
    )

    # Prefetch the next token's inputs while the current recurrent update runs.
    # The state dependency remains serialized, but its independent loads are pipelined.
    for token_offset in tl.range(
        0, sequence_length, loop_unroll_factor=1, num_stages=2
    ):
        if USE_PACKED_INPUT:
            i_t = bos + token_offset
        else:
            i_t = token_offset

        q_offsets = (
            i_b * stride_q_b + i_t * stride_q_t + i_h * stride_q_h + o_k * stride_q_k
        )
        k_offsets = (
            i_b * stride_k_b + i_t * stride_k_t + i_h * stride_k_h + o_k * stride_k_k
        )
        v_offsets = (
            i_b * stride_v_b + i_t * stride_v_t + i_hv * stride_v_h + o_v * stride_v_v
        )
        q_values = tl.load(q + q_offsets, mask=mask_k, other=0.0)
        k_values = tl.load(k + k_offsets, mask=mask_k, other=0.0)
        v_values = tl.load(v + v_offsets, mask=mask_v, other=0.0)
        q_low, k_low = _prepare_qk(
            q_values,
            k_values,
            scale,
            ACTIVATION_TYPE=q.dtype.element_ty,
            USE_QK_L2NORM=USE_QK_L2NORM,
        )

        g_offset = i_b * stride_g_b + i_t * stride_g_t + i_hv * stride_g_h
        beta_offset = i_b * stride_beta_b + i_t * stride_beta_t + i_hv * stride_beta_h
        state_acc, output = _gdn_step(
            state_acc,
            q_low,
            k_low,
            v_values,
            exp(tl.load(g + g_offset).to(tl.float32)),
            tl.load(beta + beta_offset),
            ACTIVATION_TYPE=q.dtype.element_ty,
            FP32_STATE_PRODUCTS=FP32_STATE_PRODUCTS,
            FP32_UPDATE_PRODUCTS=FP32_UPDATE_PRODUCTS,
        )
        output_offsets = (
            i_b * stride_o_b + i_t * stride_o_t + i_hv * stride_o_h + o_v * stride_o_v
        )
        tl.store(o + output_offsets, output.to(o.dtype.element_ty), mask=mask_v)

        if USE_PER_TOKEN_STATE_INDICES:
            output_state_index = tl.load(
                state_indices
                + i_n * stride_indices_seq
                + token_offset * stride_indices_tok
            ).to(tl.int64)
            output_state_valid = output_state_index >= 0
            safe_output_state_index = tl.where(
                output_state_valid, output_state_index, 0
            )
            output_state_offsets = (
                safe_output_state_index * stride_state_s
                + i_hv * stride_state_h
                + o_k[:, None] * stride_state_k
                + o_v[None, :] * stride_state_v
            )
            output_scale_offsets = (
                safe_output_state_index * stride_scale_s
                + i_hv * stride_scale_h
                + o_v * stride_scale_v
            )
            _store_dynamic_fp8_state(
                state_acc,
                state_fp8,
                state_scale,
                output_state_offsets,
                output_scale_offsets,
                mask_state & output_state_valid,
                mask_v & output_state_valid,
                ACTIVATION_TYPE=q.dtype.element_ty,
                STREAM_STATE_STORE=False,
                LOW_PRECISION_AMAX=LOW_PRECISION_AMAX,
            )

    if not USE_PER_TOKEN_STATE_INDICES:
        _store_dynamic_fp8_state(
            state_acc,
            state_fp8,
            state_scale,
            initial_state_offsets,
            initial_scale_offsets,
            mask_state,
            mask_v,
            ACTIVATION_TYPE=q.dtype.element_ty,
            STREAM_STATE_STORE=False,
            LOW_PRECISION_AMAX=LOW_PRECISION_AMAX,
        )


def quantize_gdn_state_fp8(
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize a contiguous GDN state per V channel.

    Returns the native FP8 E4M3 state and an FP32 scale tensor with shape
    ``[S, HV, V]``. Each scale covers the K elements in one state channel.
    """
    if state.ndim != 4 or not state.is_contiguous():
        raise ValueError("state must be a contiguous [S, HV, K, V] tensor")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("FP8 E4M3 is not available")

    S, HV, K, V = state.shape
    state_fp8 = torch.empty(state.shape, device=state.device, dtype=torch.float8_e4m3fn)
    state_scale = torch.empty((S, HV, V), device=state.device, dtype=torch.float32)
    use_3d_grid = S * HV > 65535
    grid = (
        (triton.cdiv(V, _STATE_BLOCK_V), HV, S)
        if use_3d_grid
        else (triton.cdiv(V, _STATE_BLOCK_V), S * HV)
    )
    _quantize_gdn_state_fp8_kernel[grid](
        state,
        state_fp8,
        state_scale,
        K=K,
        V=V,
        HV=HV,
        BK=triton.next_power_of_2(K),
        BV=_STATE_BLOCK_V,
        USE_3D_GRID=use_3d_grid,
        USE_INT64_OFFSETS=state.numel() >= 2**31,
        num_warps=4,
        num_stages=1,
    )
    return state_fp8, state_scale


def dequantize_gdn_state_fp8(
    state_fp8: torch.Tensor,
    state_scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a per-channel scaled native FP8 E4M3 GDN state."""
    if state_fp8.ndim != 4 or not state_fp8.is_contiguous():
        raise ValueError("state_fp8 must be a contiguous [S, HV, K, V] tensor")
    S, HV, K, V = state_fp8.shape
    if (
        state_scale.dtype != torch.float32
        or not state_scale.is_contiguous()
        or state_scale.shape != (S, HV, V)
    ):
        raise ValueError("state_scale must be contiguous FP32 with shape [S, HV, V]")
    state = torch.empty(state_fp8.shape, device=state_fp8.device, dtype=output_dtype)
    use_3d_grid = S * HV > 65535
    grid = (
        (triton.cdiv(V, _STATE_BLOCK_V), HV, S)
        if use_3d_grid
        else (triton.cdiv(V, _STATE_BLOCK_V), S * HV)
    )
    _dequantize_gdn_state_fp8_kernel[grid](
        state_fp8,
        state_scale,
        state,
        K=K,
        V=V,
        HV=HV,
        BK=triton.next_power_of_2(K),
        BV=_STATE_BLOCK_V,
        USE_3D_GRID=use_3d_grid,
        USE_INT64_OFFSETS=state_fp8.numel() >= 2**31,
        num_warps=4,
        num_stages=1,
    )
    return state


def _set_tma_descriptor_allocator(device: torch.device) -> None:
    """Register Triton's runtime workspace allocator for TMA descriptors."""
    if not hasattr(triton, "set_allocator"):
        return

    def _alloc(size: int, _alignment: int, _stream: int | None):
        return torch.empty((size,), dtype=torch.uint8, device=device)

    triton.set_allocator(_alloc)


def _tensor_offsets_fit_int32(tensor: torch.Tensor) -> bool:
    return (
        all(stride >= 0 for stride in tensor.stride())
        and sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
        < 2**31
    )


def _chunk_prefill_gated_delta_rule_w8a16_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_fp8: torch.Tensor,
    state_scale: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    ssm_state_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    N = B if cu_seqlens is None else cu_seqlens.numel() - 1
    BK = triton.next_power_of_2(K)
    BV = min(_STATE_BLOCK_V, triton.next_power_of_2(V))
    use_state_indices = ssm_state_indices is not None

    initial_state = torch.empty((N, HV, K, V), device=q.device, dtype=q.dtype)
    _dequantize_indexed_gdn_state_fp8_kernel[(triton.cdiv(V, BV), N * HV)](
        state_fp8,
        state_scale,
        initial_state,
        ssm_state_indices,
        stride_state_s=state_fp8.stride(0),
        stride_state_h=state_fp8.stride(1),
        stride_state_k=state_fp8.stride(2),
        stride_state_v=state_fp8.stride(3),
        stride_scale_s=state_scale.stride(0),
        stride_scale_h=state_scale.stride(1),
        stride_scale_v=state_scale.stride(2),
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_STATE_INDICES=use_state_indices,
        num_warps=4,
        num_stages=1,
    )

    q_normalized = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    k_normalized = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    _normalize_prefill_qk_kernel[(B * T * H,)](
        q,
        k,
        q_normalized,
        k_normalized,
        1.0,
        stride_q_b=q.stride(0),
        stride_q_t=q.stride(1),
        stride_q_h=q.stride(2),
        stride_q_k=q.stride(3),
        stride_k_b=k.stride(0),
        stride_k_t=k.stride(1),
        stride_k_h=k.stride(2),
        stride_k_k=k.stride(3),
        T=T,
        H=H,
        K=K,
        BK=BK,
        num_warps=4,
        num_stages=1,
    )

    _set_tma_descriptor_allocator(q.device)
    _, output, _, final_state, _, _, _ = chunk_gated_delta_rule_fwd(
        q=q_normalized,
        k=k_normalized,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    _quantize_indexed_gdn_state_fp8_kernel[(triton.cdiv(V, BV), N * HV)](
        final_state,
        state_fp8,
        state_scale,
        ssm_state_indices,
        stride_state_s=state_fp8.stride(0),
        stride_state_h=state_fp8.stride(1),
        stride_state_k=state_fp8.stride(2),
        stride_state_v=state_fp8.stride(3),
        stride_scale_s=state_scale.stride(0),
        stride_scale_h=state_scale.stride(1),
        stride_scale_v=state_scale.stride(2),
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_STATE_INDICES=use_state_indices,
        num_warps=4,
        num_stages=1,
    )
    return output, state_fp8, state_scale


def _chunk_prefill_sequence_threshold(num_sequences: int) -> int:
    if num_sequences <= 1:
        return 256
    if num_sequences <= 2:
        return 192
    if num_sequences <= 4:
        return 256
    if num_sequences <= 8:
        return (1024 + num_sequences - 1) // num_sequences
    if num_sequences <= 64:
        return (2048 + num_sequences - 1) // num_sequences
    return max(24, (2048 + num_sequences - 1) // num_sequences)


def fused_recurrent_gated_delta_rule_w8a16_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    state_fp8: torch.Tensor,
    state_scale: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    *,
    max_sequence_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run recurrent GDN with dynamically scaled FP8 state.

    Packed inputs use shape ``[1, total_tokens, H, K]`` and ``cu_seqlens``
    describes each variable-length sequence. Non-packed inputs use shape
    ``[N, T, H, K]``. Recurrent products select FP32 for sequence processing
    and small decode grids, while large decode grids use the BF16/FP16
    activation dtype. Reductions and the recurrent accumulator remain FP32.
    This is the dynamically quantized state specialization of
    ``fused_recurrent_gated_delta_rule_fwd``. ``initial_state`` is represented
    by ``state_fp8`` and its per-V-channel ``state_scale``; both tensors are
    always updated in-place and returned with the output. During speculative
    decoding, ``num_accepted_tokens`` selects the initial state slot for each
    sequence, and each candidate token writes both its FP8 state and scale to
    the matching slot in ``ssm_state_indices``. Packed decode callers may pass
    ``max_sequence_length=1`` to select the single-token fast path without
    synchronizing ``cu_seqlens`` back to CPU.
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
        if B != 1:
            raise ValueError("packed input requires shape [1, total_tokens, ...]")

    if state_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError("state_fp8 must have dtype torch.float8_e4m3fn")
    if not state_fp8.is_contiguous() or state_fp8.shape[1:] != (HV, K, V):
        raise ValueError("state_fp8 must be contiguous with shape [S, HV, K, V]")
    if (
        state_scale.dtype != torch.float32
        or not state_scale.is_contiguous()
        or state_scale.shape != (state_fp8.shape[0], HV, V)
    ):
        raise ValueError("state_scale must be contiguous FP32 with shape [S, HV, V]")
    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
        use_per_token_state_indices = False
    else:
        if ssm_state_indices.device != q.device or ssm_state_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("ssm_state_indices must be an integer tensor on q.device")
        if ssm_state_indices.ndim == 1:
            if ssm_state_indices.shape[0] != N:
                raise ValueError("ssm_state_indices must contain one row per sequence")
            stride_indices_seq = ssm_state_indices.stride(0)
            stride_indices_tok = 0
            use_per_token_state_indices = False
        elif ssm_state_indices.ndim == 2:
            if ssm_state_indices.shape[0] != N:
                raise ValueError("ssm_state_indices must contain one row per sequence")
            stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()
            use_per_token_state_indices = True
        else:
            raise ValueError("ssm_state_indices must be a 1D or 2D tensor")
    if num_accepted_tokens is not None:
        if ssm_state_indices is None or ssm_state_indices.ndim != 2:
            raise ValueError(
                "num_accepted_tokens requires 2D per-token ssm_state_indices"
            )
        if (
            num_accepted_tokens.device != q.device
            or num_accepted_tokens.dtype not in (torch.int32, torch.int64)
            or not num_accepted_tokens.is_contiguous()
            or num_accepted_tokens.shape != (N,)
        ):
            raise ValueError(
                "num_accepted_tokens must be a contiguous integer tensor "
                "with shape [N] on q.device"
            )

    single_token_per_sequence = max_sequence_length == 1 if use_packed_input else T == 1
    if use_packed_input:
        average_sequence_length = (
            max_sequence_length if max_sequence_length is not None else T // N
        )
    else:
        average_sequence_length = T
    use_chunk_prefill = (
        not single_token_per_sequence
        and not use_per_token_state_indices
        and num_accepted_tokens is None
        and use_qk_l2norm_in_kernel
        and K == 128
        and V == 128
        and average_sequence_length >= _chunk_prefill_sequence_threshold(N)
    )
    if use_chunk_prefill:
        return _chunk_prefill_gated_delta_rule_w8a16_fp8(
            q,
            k,
            v,
            g,
            beta,
            state_fp8,
            state_scale,
            scale,
            cu_seqlens,
            ssm_state_indices,
        )

    o = torch.empty(B, T, HV, V, device=q.device, dtype=q.dtype)
    BK = triton.next_power_of_2(K)
    kernel_args = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "o": o,
        "state_fp8": state_fp8,
        "state_scale": state_scale,
        "cu_seqlens": cu_seqlens,
        "state_indices": ssm_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
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
        "stride_state_s": state_fp8.stride(0),
        "stride_state_h": state_fp8.stride(1),
        "stride_state_k": state_fp8.stride(2),
        "stride_state_v": state_fp8.stride(3),
        "stride_scale_s": state_scale.stride(0),
        "stride_scale_h": state_scale.stride(1),
        "stride_scale_v": state_scale.stride(2),
        "stride_indices_seq": stride_indices_seq,
        "stride_indices_tok": stride_indices_tok,
        "H": H,
        "HV": HV,
        "K": K,
        "V": V,
        "BK": BK,
        "USE_PACKED_INPUT": use_packed_input,
        "USE_STATE_INDICES": ssm_state_indices is not None,
        "USE_SPEC_DECODING": num_accepted_tokens is not None,
        "USE_QK_L2NORM": use_qk_l2norm_in_kernel,
    }

    if not single_token_per_sequence:
        block_v = min(_STATE_BLOCK_V, triton.next_power_of_2(V))
        if K == 128 and V == 128:
            if N == 1:
                block_v = 16 if T <= 8 else 8
            elif N == 2:
                block_v = 16
            elif N <= 4 and T >= 8 * N:
                block_v = 8
        _fused_recurrent_gated_delta_rule_sequence_w8a16_fp8_kernel[
            (triton.cdiv(V, block_v), N * HV)
        ](
            **kernel_args,
            T=T,
            BV=block_v,
            USE_PER_TOKEN_STATE_INDICES=use_per_token_state_indices,
            LOW_PRECISION_AMAX=False,
            FP32_STATE_PRODUCTS=True,
            FP32_UPDATE_PRODUCTS=not (
                (N == 1 and average_sequence_length >= 16)
                or (N == 4 and average_sequence_length >= 8)
            ),
            num_warps=1,
            num_stages=2,
        )
        return o, state_fp8, state_scale

    block_v = _STATE_BLOCK_V
    if K == 128 and V == 128:
        if N == 1:
            block_v = 8
        elif N == 2:
            block_v = 16

    use_fp32_products = N < 128 and N != 2
    use_3d_decode_grid = N * HV > 65535
    num_v_tiles = triton.cdiv(V, block_v)
    use_grouped_decode = (
        K == 128 and V == 128 and HV == 2 * H and N * HV * num_v_tiles == 1024
    )
    if use_grouped_decode:
        _fused_recurrent_gated_delta_rule_grouped_w8a16_fp8_kernel[
            (num_v_tiles, N * H)
        ](
            **kernel_args,
            BV=block_v,
            LOW_PRECISION_AMAX=N >= 128,
            FP32_STATE_PRODUCTS=use_fp32_products,
            FP32_UPDATE_PRODUCTS=use_fp32_products,
            N=N,
            num_warps=1,
            num_stages=1,
        )
        return o, state_fp8, state_scale

    state_offsets_safe = N < 128 or _tensor_offsets_fit_int32(state_fp8)
    other_offsets_safe = N < 128 or all(
        _tensor_offsets_fit_int32(tensor)
        for tensor in (q, k, v, g, beta, o, state_scale)
    )
    use_tma_load = (
        N >= 1024 and K == 128 and V == 128 and other_offsets_safe and is_tma_supported
    )
    if use_tma_load:
        tma_grid = (num_v_tiles, HV, N) if use_3d_decode_grid else (num_v_tiles, N * HV)
        _set_tma_descriptor_allocator(q.device)
        _fused_recurrent_gated_delta_rule_w8a16_fp8_kernel[tma_grid](
            **kernel_args,
            S=state_fp8.shape[0],
            BV=block_v,
            USE_3D_GRID=use_3d_decode_grid,
            USE_TMA_LOAD=True,
            USE_INT64_STATE_OFFSETS=not state_offsets_safe,
            LOW_PRECISION_AMAX=N >= 128,
            FP32_STATE_PRODUCTS=use_fp32_products,
            FP32_UPDATE_PRODUCTS=use_fp32_products,
            N=N,
            num_warps=1,
            num_stages=1,
        )
        return o, state_fp8, state_scale

    if not other_offsets_safe:
        _fused_recurrent_gated_delta_rule_sequence_w8a16_fp8_kernel[
            (num_v_tiles, N * HV)
        ](
            **kernel_args,
            T=T,
            BV=block_v,
            USE_PER_TOKEN_STATE_INDICES=use_per_token_state_indices,
            LOW_PRECISION_AMAX=True,
            FP32_STATE_PRODUCTS=False,
            FP32_UPDATE_PRODUCTS=False,
            num_warps=1,
            num_stages=2,
        )
        return o, state_fp8, state_scale

    decode_grid = (
        (triton.cdiv(V, block_v), HV, N)
        if use_3d_decode_grid
        else (triton.cdiv(V, block_v), N * HV)
    )
    _fused_recurrent_gated_delta_rule_w8a16_fp8_kernel[decode_grid](
        **kernel_args,
        S=state_fp8.shape[0],
        BV=block_v,
        USE_3D_GRID=use_3d_decode_grid,
        USE_TMA_LOAD=False,
        USE_INT64_STATE_OFFSETS=not state_offsets_safe,
        LOW_PRECISION_AMAX=N >= 128,
        FP32_STATE_PRODUCTS=use_fp32_products,
        FP32_UPDATE_PRODUCTS=use_fp32_products,
        N=N,
        num_warps=1,
        num_stages=1,
    )
    return o, state_fp8, state_scale
