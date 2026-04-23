import logging
import math

import torch
import triton.language as tl

logger = logging.getLogger(__name__)


def scaled_dot_product_attention_forward(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    logger.debug("GEMS GCU400 SDPA FORWARD (decomposed)")

    HEAD_DIM_K = query.shape[-1]
    assert dropout_p == 0.0, "Currently only support dropout_p=0.0"

    if scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM_K)
    else:
        sm_scale = scale

    q_head_num = query.shape[1]
    kv_head_num = key.shape[1]

    if enable_gqa and q_head_num != kv_head_num:
        group = q_head_num // kv_head_num
        key = key.repeat_interleave(group, dim=1)
        value = value.repeat_interleave(group, dim=1)

    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * sm_scale

    if is_causal:
        q_len = query.shape[2]
        kv_len = key.shape[2]
        causal_mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=query.device).triu(diagonal=kv_len - q_len + 1)
        attn_weight = attn_weight.masked_fill(causal_mask, float("-inf"))

    if attn_mask is not None:
        attn_weight = attn_weight + attn_mask

    attn_weight = torch.softmax(attn_weight, dim=-1)
    output = torch.matmul(attn_weight, value)

    M = torch.empty(
        (query.shape[0], query.shape[1], query.shape[2]),
        device=query.device,
        dtype=torch.float32,
    )

    return output, M


def scaled_dot_product_attention_backward(
    do,
    query,
    key,
    value,
    o,
    M,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    logger.debug("GEMS GCU400 SDPA BACKWARD (decomposed)")
    from flag_gems.ops.attention import scaled_dot_product_attention_backward as generic_backward
    return generic_backward(do, query, key, value, o, M, attn_mask, dropout_p, is_causal, scale, enable_gqa)


class ScaleDotProductAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        o, M = scaled_dot_product_attention_forward(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )
        ctx.save_for_backward(query, key, value, o, M)
        ctx.attn_mask = attn_mask
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.enable_gqa = enable_gqa
        return o

    @staticmethod
    def backward(ctx, do):
        query, key, value, o, M = ctx.saved_tensors
        dq, dk, dv = scaled_dot_product_attention_backward(
            do,
            query,
            key,
            value,
            o,
            M,
            ctx.attn_mask,
            ctx.dropout_p,
            ctx.is_causal,
            ctx.scale,
            ctx.enable_gqa,
        )
        return dq, dk, dv, None, None, None, None, None


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    return ScaleDotProductAttention.apply(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    )
