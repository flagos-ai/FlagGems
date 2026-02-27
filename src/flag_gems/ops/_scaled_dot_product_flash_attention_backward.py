import logging
import math

import torch

from flag_gems.ops.attention import scaled_dot_product_attention_backward

logger = logging.getLogger(__name__)


def _scaled_dot_product_flash_attention_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    *,
    scale=None,
):
    """
    Backward pass for scaled dot product flash attention.

    This function computes gradients for query, key, and value tensors given the
    gradient of the output from the forward pass.

    Args:
        grad_out: Gradient of the output tensor from the forward pass.
                  Shape: [batch, num_heads, seq_len_q, head_dim]
        query: Query tensor from the forward pass.
               Shape: [batch, num_heads, seq_len_q, head_dim]
        key: Key tensor from the forward pass.
             Shape: [batch, num_heads, seq_len_k, head_dim]
        value: Value tensor from the forward pass.
               Shape: [batch, num_heads, seq_len_k, head_dim]
        out: Output tensor from the forward pass.
             Shape: [batch, num_heads, seq_len_q, head_dim]
        logsumexp: Log-sum-exp values from the forward pass (natural log).
                   Shape: [batch, num_heads, seq_len_q]
        cum_seq_q: Cumulative sequence lengths for queries (for variable length).
        cum_seq_k: Cumulative sequence lengths for keys (for variable length).
        max_q: Maximum query sequence length.
        max_k: Maximum key sequence length.
        dropout_p: Dropout probability.
        is_causal: Whether to use causal masking.
        philox_seed: RNG seed for dropout.
        philox_offset: RNG offset for dropout.
        scale: Optional scaling factor for attention scores.

    Returns:
        Tuple of (grad_query, grad_key, grad_value)
    """
    logger.debug("GEMS _SCALED_DOT_PRODUCT_FLASH_ATTENTION_BACKWARD")

    # Currently only supports the non-variable-length case
    assert cum_seq_q is None or cum_seq_q.numel() == 0, (
        "Variable-length sequences not supported yet"
    )
    assert cum_seq_k is None or cum_seq_k.numel() == 0, (
        "Variable-length sequences not supported yet"
    )
    assert dropout_p == 0.0, "Dropout is not supported yet in backward pass"

    # Shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    # Compute scale
    if scale is None:
        sm_scale = 1.0 / (HEAD_DIM_K**0.5)
    else:
        sm_scale = scale

    # Convert logsumexp from natural log to log2 format
    # The backward kernel expects M = logsumexp * log2(e) = logsumexp / ln(2)
    RCP_LN2 = 1.0 / math.log(2)
    M = logsumexp * RCP_LN2

    # Call the existing backward implementation
    dq, dk, dv = scaled_dot_product_attention_backward(
        grad_out,
        query,
        key,
        value,
        out,
        M,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=sm_scale,
        enable_gqa=(query.shape[1] != key.shape[1]),
    )

    return dq, dk, dv
