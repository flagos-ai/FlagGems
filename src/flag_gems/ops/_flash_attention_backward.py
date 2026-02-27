import logging

import torch
import torch.nn.functional as F

from flag_gems.ops.attention import scaled_dot_product_attention_backward

logger = logging.getLogger(__name__)


def _flash_attention_backward(
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
    window_size_left=None,
    window_size_right=None,
):
    """
    Flash Attention backward pass.

    Args:
        grad_out: Gradient of output tensor
        query: Query tensor (batch, seqlen_q, num_heads, head_dim)
        key: Key tensor (batch, seqlen_k, num_heads_k, head_dim)
        value: Value tensor (batch, seqlen_k, num_heads_k, head_dim)
        out: Forward output tensor
        logsumexp: Log-sum-exp from forward pass
        cum_seq_q: Cumulative sequence lengths for query (for varlen attention)
        cum_seq_k: Cumulative sequence lengths for key (for varlen attention)
        max_q: Maximum query sequence length
        max_k: Maximum key sequence length
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
        philox_seed: Random seed for dropout
        philox_offset: Random offset for dropout
        scale: Optional scale factor (default: 1/sqrt(head_dim))
        window_size_left: Left window size for sliding window attention
        window_size_right: Right window size for sliding window attention

    Returns:
        Tuple of (grad_query, grad_key, grad_value)
    """
    logger.debug("GEMS _FLASH_ATTENTION_BACKWARD")

    # Check that varlen is not used (cum_seq_q and cum_seq_k should be empty or None)
    # The flash_attention_backward in ATen expects tensors for cum_seq but they may be empty
    has_varlen = (cum_seq_q is not None and cum_seq_q.numel() > 0) or (
        cum_seq_k is not None and cum_seq_k.numel() > 0
    )

    if has_varlen:
        # For varlen attention, we need special handling
        # Currently not fully supported, raise an error
        raise NotImplementedError(
            "Variable length flash attention backward is not yet supported in FlagGems. "
            "Please use standard attention shapes."
        )

    # Input shapes for flash attention:
    # query: (batch, seqlen_q, num_heads, head_dim)
    # key/value: (batch, seqlen_k, num_heads_k, head_dim)
    # We need to transpose to (batch, num_heads, seqlen, head_dim) for our backend
    HEAD_DIM = query.shape[-1]

    # Validate inputs
    assert dropout_p == 0.0, "Dropout in flash attention backward is not yet supported"

    # Ensure tensors are contiguous for proper memory access
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    out = out.contiguous()
    grad_out = grad_out.contiguous()

    # Transpose from (batch, seqlen, heads, dim) to (batch, heads, seqlen, dim)
    query_t = query.transpose(1, 2).contiguous()
    key_t = key.transpose(1, 2).contiguous()
    value_t = value.transpose(1, 2).contiguous()
    out_t = out.transpose(1, 2).contiguous()
    grad_out_t = grad_out.transpose(1, 2).contiguous()

    # Handle logsumexp shape - it may be (batch, heads, seqlen_q) or (heads, total_seqlen)
    # The backend expects (batch, heads, seqlen_q)
    if logsumexp.dim() == 2:
        # Shape: (heads, total_seqlen) - reshape to (batch, heads, seqlen_q)
        batch_size = query.shape[0]
        num_heads = query.shape[2]
        seqlen_q = query.shape[1]
        M = logsumexp.view(batch_size, num_heads, seqlen_q)
    else:
        # Shape should be (batch, heads, seqlen_q)
        M = logsumexp

    # Compute gradients using the backend function
    dq, dk, dv = scaled_dot_product_attention_backward(
        grad_out_t,
        query_t,
        key_t,
        value_t,
        out_t,
        M,
        attn_mask=None,
        dropout_p=0.0,  # dropout handled separately in flash attention
        is_causal=is_causal,
        scale=scale,
        enable_gqa=(query.shape[2] != key.shape[2]),  # GQA if heads differ
    )

    # Transpose gradients back from (batch, heads, seqlen, dim) to (batch, seqlen, heads, dim)
    grad_query = dq.transpose(1, 2).contiguous()
    grad_key = dk.transpose(1, 2).contiguous()
    grad_value = dv.transpose(1, 2).contiguous()

    return grad_query, grad_key, grad_value
