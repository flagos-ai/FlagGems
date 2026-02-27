import logging
import math

import torch

from flag_gems.ops.attention import scaled_dot_product_attention_forward

logger = logging.getLogger(__name__)

# ln(2) constant for converting log2 to natural log
LN2 = math.log(2)


def _scaled_dot_product_flash_attention(
    query,
    key,
    value,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    *,
    scale=None,
):
    """
    FlagGems implementation of scaled dot product flash attention.

    This wraps the scaled_dot_product_attention_forward function to match the PyTorch
    _scaled_dot_product_flash_attention aten op signature.

    Args:
        query: (batch, num_heads, seq_len_q, head_dim)
        key: (batch, num_heads, seq_len_k, head_dim)
        value: (batch, num_heads, seq_len_k, head_dim)
        dropout_p: Dropout probability (default: 0.0)
        is_causal: Whether to apply causal masking (default: False)
        return_debug_mask: Whether to return debug attention mask (default: False)
        scale: Optional scale factor (default: None, uses 1/sqrt(head_dim))

    Returns:
        Tuple of 9 outputs:
        - output: (batch, num_heads, seq_len_q, head_dim)
        - logsumexp: (batch, num_heads, seq_len_q)
        - cum_seq_q: None (not used for non-varlen)
        - cum_seq_k: None (not used for non-varlen)
        - max_q: seq_len_q
        - max_k: seq_len_k
        - philox_seed: RNG seed tensor (scalar on CPU)
        - philox_offset: RNG offset tensor (scalar on CPU)
        - debug_attn_mask: Empty tensor or debug mask
    """
    logger.debug("GEMS _SCALED_DOT_PRODUCT_FLASH_ATTENTION")

    # Get sequence lengths
    seq_len_q = query.shape[2]
    seq_len_k = key.shape[2]

    # Call scaled_dot_product_attention_forward
    # Returns: (output, logsumexp)
    # Note: The logsumexp is in log2 space, need to convert to natural log
    out, lse = scaled_dot_product_attention_forward(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=False,
    )

    # Convert logsumexp from log2 to natural log (multiply by ln(2))
    # The internal implementation uses log2 for numerical stability
    if lse is not None:
        lse = lse * LN2

    # Create placeholder philox tensors for RNG state
    # These are used for reproducibility with dropout, but since we don't support dropout yet,
    # we just create placeholder values
    philox_seed = torch.tensor(0, dtype=torch.int64, device="cpu")
    philox_offset = torch.tensor(0, dtype=torch.int64, device="cpu")

    # Create empty debug mask tensor
    debug_mask = torch.empty(0, dtype=query.dtype, device=query.device)

    # Return 9 outputs matching the aten op signature
    # cum_seq_q and cum_seq_k are None for non-varlen attention
    return (
        out,
        lse,
        None,  # cum_seq_q
        None,  # cum_seq_k
        seq_len_q,  # max_q
        seq_len_k,  # max_k
        philox_seed,
        philox_offset,
        debug_mask,
    )
