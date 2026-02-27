import logging

import torch

from flag_gems.ops.attention import scaled_dot_product_attention_forward

logger = logging.getLogger(__name__)


def _scaled_dot_product_cudnn_attention(
    query,
    key,
    value,
    attn_bias=None,
    compute_log_sumexp=True,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    *,
    scale=None,
):
    """
    Scaled dot product attention using FlagGems Triton kernels.

    This function wraps the FlagGems attention implementation to provide
    the same interface as torch._scaled_dot_product_cudnn_attention.

    Args:
        query: Query tensor of shape (batch, num_heads, seq_len_q, head_dim)
        key: Key tensor of shape (batch, num_heads, seq_len_k, head_dim)
        value: Value tensor of shape (batch, num_heads, seq_len_k, head_dim)
        attn_bias: Optional attention bias tensor
        compute_log_sumexp: Whether to compute and return log-sum-exp values
        dropout_p: Dropout probability (currently must be 0.0)
        is_causal: Whether to apply causal masking
        return_debug_mask: Whether to return debug attention mask
        scale: Optional scale factor for attention scores

    Returns:
        tuple: (output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
                philox_seed, philox_offset, debug_attn_mask)
    """
    logger.debug("GEMS _SCALED_DOT_PRODUCT_CUDNN_ATTENTION")

    # Convert attn_bias to attn_mask if provided
    attn_mask = attn_bias

    # Call the existing FlagGems attention forward implementation
    output, M = scaled_dot_product_attention_forward(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=False,
    )

    # Compute logsumexp from M if needed
    # M is already the log-sum-exp values from the attention computation
    if compute_log_sumexp:
        # M has shape (batch, num_heads, seq_len_q)
        logsumexp = M
    else:
        logsumexp = torch.empty(0, device=query.device, dtype=torch.float32)

    # Create placeholder tensors for cumulative sequence lengths
    # These are used for variable-length sequences, which we don't handle here
    cum_seq_q = torch.empty(0, device=query.device, dtype=torch.int32)
    cum_seq_k = torch.empty(0, device=query.device, dtype=torch.int32)

    # Get max sequence lengths
    max_q = query.shape[2]
    max_k = key.shape[2]

    # Philox seed and offset for dropout (currently not used as dropout_p must be 0)
    philox_seed = torch.empty((), device=query.device, dtype=torch.int64)
    philox_offset = torch.empty((), device=query.device, dtype=torch.int64)

    # Debug attention mask
    if return_debug_mask:
        # Create a debug mask of the appropriate shape
        batch = query.shape[0]
        num_heads = query.shape[1]
        seq_len_q = query.shape[2]
        seq_len_k = key.shape[2]
        debug_attn_mask = torch.zeros(
            (batch, num_heads, seq_len_q, seq_len_k),
            device=query.device,
            dtype=query.dtype,
        )
    else:
        debug_attn_mask = torch.empty(0, device=query.device, dtype=query.dtype)

    return (
        output,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        philox_seed,
        philox_offset,
        debug_attn_mask,
    )
