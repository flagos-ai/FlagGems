import logging
import math

import torch

logger = logging.getLogger(__name__)


def _scaled_dot_product_attention_math(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    """
    Computes scaled dot-product attention using explicit math operations.

    Args:
        query: Query tensor of shape (batch, heads, seq_len, head_dim)
        key: Key tensor of shape (batch, heads, seq_len, head_dim)
        value: Value tensor of shape (batch, heads, seq_len, head_dim)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (not supported, must be 0.0)
        is_causal: Whether to apply causal masking
        scale: Scale factor (default: 1/sqrt(head_dim))

    Returns:
        Tuple of (output, attention_weights)
    """
    logger.debug("GEMS _SCALED_DOT_PRODUCT_ATTENTION_MATH")

    # Get dimensions
    batch, heads, q_seq_len, head_dim = query.shape
    _, _, kv_seq_len, _ = key.shape

    # Compute scale factor
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Compute attention scores: Q @ K^T * scale
    # query: (batch, heads, q_seq_len, head_dim)
    # key.transpose(-2, -1): (batch, heads, head_dim, kv_seq_len)
    # attn_weights: (batch, heads, q_seq_len, kv_seq_len)
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply attention mask if provided
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # Boolean mask: True means "keep", False means "mask out"
            attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
        else:
            # Additive mask
            attn_weights = attn_weights + attn_mask

    # Apply causal mask if needed
    if is_causal:
        # Create causal mask: positions can only attend to earlier positions
        causal_mask = torch.ones(
            q_seq_len, kv_seq_len, dtype=torch.bool, device=query.device
        ).tril(diagonal=kv_seq_len - q_seq_len)
        attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))

    # Apply softmax to get attention probabilities
    attn_weights = torch.softmax(attn_weights, dim=-1)

    # Note: dropout is not implemented for simplicity
    # The math implementation in PyTorch also typically doesn't apply dropout
    # to the returned attention weights

    # Compute output: attn_weights @ V
    # attn_weights: (batch, heads, q_seq_len, kv_seq_len)
    # value: (batch, heads, kv_seq_len, head_dim)
    # output: (batch, heads, q_seq_len, head_dim)
    output = torch.matmul(attn_weights, value)

    return output, attn_weights
