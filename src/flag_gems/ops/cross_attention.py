import logging

import torch

import flag_gems

logger = logging.getLogger(__name__)


def cross_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
) -> torch.Tensor:
    """
    Cross-attention operator for FlagGems.

    Computes scaled dot-product attention where query comes from one sequence
    (e.g., decoder) while key and value come from another sequence (e.g., encoder).

    This is the standard cross-attention mechanism used in encoder-decoder transformer
    architectures like T5, BART, etc.

    This implementation delegates to scaled_dot_product_attention which already
    handles cross-attention scenarios correctly (Q from one source, KV from another).

    Args:
        query: Query tensor of shape (batch, head, q_seq, head_dim)
        key: Key tensor of shape (batch, kv_head, kv_seq, head_dim)
        value: Value tensor of shape (batch, kv_head, kv_seq, head_dim)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (currently only 0.0 supported)
        is_causal: Whether to apply causal masking
        scale: Optional scale factor for QK^T

    Returns:
        Output tensor of shape (batch, head, q_seq, head_dim)
    """
    logger.debug("GEMS CROSS_ATTENTION")

    # Delegate to scaled_dot_product_attention which handles cross-attention correctly
    # scaled_dot_product_attention already takes Q, K, V as separate inputs and
    # works correctly when Q and KV have different sequence lengths
    return flag_gems.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=False,
    )
