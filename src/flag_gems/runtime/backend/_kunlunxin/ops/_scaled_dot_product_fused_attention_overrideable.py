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

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _sdpa_score(
    q_ptr,
    k_ptr,
    q_row,
    k_row,
    q_stride_s,
    q_stride_d,
    k_stride_s,
    k_stride_d,
    HEAD_DIM: tl.constexpr,
):
    score = 0.0
    for d in range(0, HEAD_DIM):
        q = tl.load(q_ptr + q_row * q_stride_s + d * q_stride_d).to(tl.float32)
        k = tl.load(k_ptr + k_row * k_stride_s + d * k_stride_d).to(tl.float32)
        score += q * k
    return score


@triton.jit
def _sdpa_forward_kernel(
    query,
    key,
    value,
    bias,
    output,
    logsumexp,
    q_stride_b,
    q_stride_h,
    q_stride_s,
    q_stride_d,
    k_stride_b,
    k_stride_h,
    k_stride_s,
    k_stride_d,
    v_stride_b,
    v_stride_h,
    v_stride_s,
    v_stride_d,
    o_stride_b,
    o_stride_h,
    o_stride_s,
    o_stride_d,
    lse_stride_b,
    lse_stride_h,
    lse_stride_s,
    bias_stride_b,
    bias_stride_h,
    bias_stride_q,
    bias_stride_k,
    SEQ_Q: tl.constexpr,
    H_Q: tl.constexpr,
    H_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEQ_K: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BIAS_IS_BOOL: tl.constexpr,
):
    pid = tl.program_id(0)
    q_row = pid % SEQ_Q
    q_head = (pid // SEQ_Q) % H_Q
    batch = pid // (SEQ_Q * H_Q)
    kv_head = q_head // (H_Q // H_K)

    q_base = query + batch * q_stride_b + q_head * q_stride_h
    k_base = key + batch * k_stride_b + kv_head * k_stride_h
    v_base = value + batch * v_stride_b + kv_head * v_stride_h

    max_score = -float("inf")
    for k_row in range(0, SEQ_K):
        score = _sdpa_score(
            q_base,
            k_base,
            q_row,
            k_row,
            q_stride_s,
            q_stride_d,
            k_stride_s,
            k_stride_d,
            HEAD_DIM,
        ) * SOFTMAX_SCALE
        if HAS_BIAS:
            bias_value = tl.load(
                bias
                + batch * bias_stride_b
                + q_head * bias_stride_h
                + q_row * bias_stride_q
                + k_row * bias_stride_k
            )
            if BIAS_IS_BOOL:
                score = tl.where(bias_value, score, -float("inf"))
            else:
                score += bias_value.to(tl.float32)
        if IS_CAUSAL:
            score = tl.where(k_row <= q_row, score, -float("inf"))
        max_score = tl.maximum(max_score, score)

    denom = 0.0
    for k_row in range(0, SEQ_K):
        score = _sdpa_score(
            q_base,
            k_base,
            q_row,
            k_row,
            q_stride_s,
            q_stride_d,
            k_stride_s,
            k_stride_d,
            HEAD_DIM,
        ) * SOFTMAX_SCALE
        if HAS_BIAS:
            bias_value = tl.load(
                bias
                + batch * bias_stride_b
                + q_head * bias_stride_h
                + q_row * bias_stride_q
                + k_row * bias_stride_k
            )
            if BIAS_IS_BOOL:
                score = tl.where(bias_value, score, -float("inf"))
            else:
                score += bias_value.to(tl.float32)
        if IS_CAUSAL:
            score = tl.where(k_row <= q_row, score, -float("inf"))
        denom += tl.exp(score - max_score)

    for d in range(0, HEAD_DIM):
        acc = 0.0
        for k_row in range(0, SEQ_K):
            score = _sdpa_score(
                q_base,
                k_base,
                q_row,
                k_row,
                q_stride_s,
                q_stride_d,
                k_stride_s,
                k_stride_d,
                HEAD_DIM,
            ) * SOFTMAX_SCALE
            if HAS_BIAS:
                bias_value = tl.load(
                    bias
                    + batch * bias_stride_b
                    + q_head * bias_stride_h
                    + q_row * bias_stride_q
                    + k_row * bias_stride_k
                )
                if BIAS_IS_BOOL:
                    score = tl.where(bias_value, score, -float("inf"))
                else:
                    score += bias_value.to(tl.float32)
            if IS_CAUSAL:
                score = tl.where(k_row <= q_row, score, -float("inf"))
            probability = tl.exp(score - max_score) / denom
            value_element = tl.load(v_base + k_row * v_stride_s + d * v_stride_d).to(
                tl.float32
            )
            acc += probability * value_element
        tl.store(
            output
            + batch * o_stride_b
            + q_head * o_stride_h
            + q_row * o_stride_s
            + d * o_stride_d,
            acc,
        )

    tl.store(
        logsumexp + batch * lse_stride_b + q_head * lse_stride_h + q_row * lse_stride_s,
        max_score + tl.log(denom),
    )


@triton.jit
def _sdpa_init_metadata_kernel(
    cum_seq_q,
    cum_seq_k,
    philox_seed,
    philox_offset,
    seq_q,
    seq_k,
    BATCH: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < BATCH:
        tl.store(cum_seq_q + pid * 2, 0)
        tl.store(cum_seq_q + pid * 2 + 1, seq_q)
        tl.store(cum_seq_k + pid * 2, 0)
        tl.store(cum_seq_k + pid * 2 + 1, seq_k)
    if pid == 0:
        tl.store(philox_seed, 0)
        tl.store(philox_offset, 0)


def _bias_strides(attn_bias, batch_size, num_heads, seq_q, seq_k):
    if attn_bias is None:
        return (0, 0, 0, 0), False
    if attn_bias.ndim == 2:
        shape = (1, 1, *attn_bias.shape)
        strides = (0, 0, *attn_bias.stride())
    elif attn_bias.ndim == 3:
        shape = (attn_bias.shape[0], 1, *attn_bias.shape[1:])
        strides = (attn_bias.stride(0), 0, *attn_bias.stride()[1:])
    elif attn_bias.ndim == 4:
        shape = tuple(attn_bias.shape)
        strides = attn_bias.stride()
    else:
        raise RuntimeError("attn_bias must be a dense 2D, 3D, or 4D tensor")

    expected = (batch_size, num_heads, seq_q, seq_k)
    if any(actual not in (1, wanted) for actual, wanted in zip(shape, expected)):
        raise RuntimeError("attn_bias must broadcast to (batch, query_heads, query_seq, key_seq)")
    return tuple(stride if size != 1 else 0 for size, stride in zip(shape, strides)), True


def _scaled_dot_product_fused_attention_overrideable(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: float = None,
):
    logger.debug("GEMS_KUNLUNXIN SCALED DOT PRODUCT FUSED ATTENTION OVERRIDEABLE")
    assert dropout_p == 0.0, "Only dropout_p=0.0 is supported"
    assert query.ndim == key.ndim == value.ndim == 4, "Only dense 4D attention is supported"
    assert query.shape[0] == key.shape[0] == value.shape[0], "Batch dimensions must match"
    assert key.shape == value.shape, "key and value must have matching shapes"
    assert query.shape[-1] == key.shape[-1] in (64, 128), "Only head dimensions 64 and 128 are supported"
    assert query.shape[1] % key.shape[1] == 0, "Query heads must be a multiple of key heads"

    batch_size, num_query_heads, seq_q, head_dim = query.shape
    _, num_key_heads, seq_k, _ = key.shape
    bias_strides, has_bias = _bias_strides(
        attn_bias, batch_size, num_query_heads, seq_q, seq_k
    )
    if has_bias:
        assert attn_bias.device == query.device, "attn_bias must be on the query device"

    output = torch.empty_like(query)
    logsumexp = torch.empty(
        (batch_size, num_query_heads, seq_q), dtype=torch.float32, device=query.device
    )
    cum_seq_q = torch.empty((batch_size, 2), dtype=torch.int32, device=query.device)
    cum_seq_k = torch.empty((batch_size, 2), dtype=torch.int32, device=query.device)
    philox_seed = torch.empty(1, dtype=torch.int64, device=query.device)
    philox_offset = torch.empty(1, dtype=torch.int64, device=query.device)
    debug_attn_mask = (
        torch.empty(
            (batch_size, num_query_heads, seq_q, seq_k),
            dtype=query.dtype,
            device=query.device,
        )
        if return_debug_mask
        else torch.empty(0, dtype=query.dtype, device=query.device)
    )

    _sdpa_init_metadata_kernel[(max(batch_size, 1),)](
        cum_seq_q,
        cum_seq_k,
        philox_seed,
        philox_offset,
        seq_q,
        seq_k,
        BATCH=batch_size,
        num_warps=1,
    )
    _sdpa_forward_kernel[(batch_size * num_query_heads * seq_q,)](
        query,
        key,
        value,
        attn_bias if has_bias else query,
        output,
        logsumexp,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        *logsumexp.stride(),
        *bias_strides,
        SEQ_Q=seq_q,
        H_Q=num_query_heads,
        H_K=num_key_heads,
        HEAD_DIM=head_dim,
        SEQ_K=seq_k,
        SOFTMAX_SCALE=scale if scale is not None else head_dim**-0.5,
        IS_CAUSAL=is_causal,
        HAS_BIAS=has_bias,
        BIAS_IS_BOOL=has_bias and attn_bias.dtype == torch.bool,
        num_warps=1,
    )
    return (
        output,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        seq_q,
        seq_k,
        philox_seed,
        philox_offset,
        debug_attn_mask,
    )
