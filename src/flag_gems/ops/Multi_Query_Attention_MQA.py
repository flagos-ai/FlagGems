import logging

import torch
import torch.nn.functional as F

from flag_gems.ops.attention import scaled_dot_product_attention

logger = logging.getLogger(__name__)


def multi_query_attention_mqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
) -> torch.Tensor:
    """Multi-Query Attention (MQA) implementation.

    In Multi-Query Attention, the key and value tensors have a single head
    (i.e., num_key_heads = num_value_heads = 1), while the query tensor
    can have multiple heads. This is a special case of Grouped-Query Attention (GQA).

    Args:
        query: Query tensor of shape (batch, num_query_heads, seq_len, head_dim)
        key: Key tensor of shape (batch, 1, kv_seq_len, head_dim) - must have 1 head
        value: Value tensor of shape (batch, 1, kv_seq_len, head_dim) - must have 1 head
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (currently only 0.0 is supported)
        is_causal: Whether to apply causal masking
        scale: Optional scale factor (defaults to 1/sqrt(head_dim))

    Returns:
        Output tensor of shape (batch, num_query_heads, seq_len, head_dim)
    """
    logger.debug("GEMS MULTI_QUERY_ATTENTION_MQA")

    # Validate inputs for MQA (key and value must have 1 head)
    if key.shape[1] != 1:
        raise ValueError(
            f"Multi-Query Attention requires key to have 1 head, got {key.shape[1]}"
        )
    if value.shape[1] != 1:
        raise ValueError(
            f"Multi-Query Attention requires value to have 1 head, got {value.shape[1]}"
        )

    # Currently only dropout_p=0.0 is supported
    if dropout_p != 0.0:
        raise ValueError("Currently only dropout_p=0.0 is supported")

    # Use scaled_dot_product_attention with enable_gqa=True for MQA
    # GQA with 1 KV head is equivalent to MQA
    output = scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=True,
    )

    return output
