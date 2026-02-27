import logging
import math

import torch

from flag_gems.ops.attention import scaled_dot_product_attention_backward

logger = logging.getLogger(__name__)

# log2(e) constant for converting between natural log and log2
LOG2E = math.log2(math.e)  # 1.4426950408889634


def _scaled_dot_product_cudnn_attention_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    attn_bias: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    *,
    scale: float = None,
):
    """
    Backward pass for scaled dot product CUDNN attention.

    This function computes gradients for query, key, and value tensors
    given the gradient of the output.

    Args:
        grad_out: Gradient of the output tensor
        query: Query tensor (batch, num_heads, seq_len_q, head_dim)
        key: Key tensor (batch, num_heads, seq_len_k, head_dim)
        value: Value tensor (batch, num_heads, seq_len_k, head_dim)
        out: Output tensor from the forward pass
        logsumexp: Log-sum-exp values from the forward pass (for numerical stability)
        philox_seed: Random seed for dropout (unused in current implementation)
        philox_offset: Random offset for dropout (unused in current implementation)
        attn_bias: Attention bias tensor (can be None)
        cum_seq_q: Cumulative sequence lengths for queries (unused in non-varlen attention)
        cum_seq_k: Cumulative sequence lengths for keys (unused in non-varlen attention)
        max_q: Maximum query sequence length
        max_k: Maximum key sequence length
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
        scale: Scaling factor for attention (default: 1/sqrt(head_dim))

    Returns:
        Tuple of (grad_query, grad_key, grad_value)
    """
    logger.debug("GEMS _SCALED_DOT_PRODUCT_CUDNN_ATTENTION_BACKWARD")

    # Ensure tensors are contiguous
    grad_out = grad_out.contiguous()
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    out = out.contiguous()

    # Convert CUDNN logsumexp (natural log) to FlagGems M format (log2)
    # FlagGems uses log2 internally for numerical stability
    # M_flaggems = logsumexp_cudnn * log2(e)
    M = logsumexp * LOG2E

    # Call the existing backward implementation
    # The converted M tensor is the softmax normalization factor in log2 format
    grad_query, grad_key, grad_value = scaled_dot_product_attention_backward(
        grad_out,
        query,
        key,
        value,
        out,
        M,
        attn_mask=attn_bias if attn_bias is not None and attn_bias.numel() > 0 else None,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=False,
    )

    return grad_query, grad_key, grad_value
