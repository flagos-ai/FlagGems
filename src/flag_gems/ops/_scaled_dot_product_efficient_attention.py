import logging
import math

import torch

from flag_gems.ops.attention import scaled_dot_product_attention_forward

logger = logging.getLogger(__name__)


def _scaled_dot_product_efficient_attention(
    query,
    key,
    value,
    attn_bias=None,
    compute_log_sumexp=True,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
):
    """
    Scaled dot product efficient attention implementation.

    Args:
        query: (batch, num_heads, seq_len, head_dim)
        key: (batch, num_heads, seq_len, head_dim)
        value: (batch, num_heads, seq_len, head_dim)
        attn_bias: Optional attention bias tensor (batch, num_heads, seq_len, seq_len)
        compute_log_sumexp: Whether to compute and return log_sumexp
        dropout_p: Dropout probability (default 0.0)
        is_causal: Whether to apply causal masking (default False)
        scale: Optional scale factor for attention scores

    Returns:
        Tuple of (output, log_sumexp, philox_seed, philox_offset)
        - output: (batch, num_heads, seq_len, head_dim)
        - log_sumexp: (batch, num_heads, seq_len) if compute_log_sumexp else empty
        - philox_seed: Scalar tensor for dropout random seed
        - philox_offset: Scalar tensor for dropout random offset
    """
    logger.debug("GEMS _SCALED_DOT_PRODUCT_EFFICIENT_ATTENTION")

    # Compute scale if not provided
    head_dim = query.shape[-1]
    if scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    else:
        sm_scale = scale

    # Call the existing forward implementation
    # The existing implementation uses attn_mask parameter which is equivalent to attn_bias
    output, M = scaled_dot_product_attention_forward(
        query,
        key,
        value,
        attn_mask=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=sm_scale,
        enable_gqa=False,
    )

    # Handle log_sumexp output
    # M is computed as m_i + log2(l_i) in the kernel, we need to convert to natural log
    # M = m_i + log2(l_i), where m_i is max(qk_scale * qk * log2(e))
    # The actual log_sumexp in natural log is: M / log2(e)
    LN2 = math.log(2)
    if compute_log_sumexp:
        # Convert from log2 to natural log
        log_sumexp = M * LN2
    else:
        # Return empty tensor when compute_log_sumexp is False
        batch, num_heads, seq_len = M.shape
        log_sumexp = torch.empty(
            (batch, num_heads, 0), dtype=torch.float32, device=query.device
        )

    # Create philox seed and offset tensors for dropout
    # Since we currently only support dropout_p=0.0, these are placeholder values
    philox_seed = torch.tensor(0, dtype=torch.int64, device=query.device)
    philox_offset = torch.tensor(0, dtype=torch.int64, device=query.device)

    return output, log_sumexp, philox_seed, philox_offset
