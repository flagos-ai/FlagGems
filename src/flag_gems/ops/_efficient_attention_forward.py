import logging

import torch

from flag_gems.ops.attention import scaled_dot_product_attention_forward

logger = logging.getLogger(__name__)


def _efficient_attention_forward(
    query,
    key,
    value,
    bias=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p=0.0,
    custom_mask_type=0,
    compute_log_sumexp=False,
    *,
    scale=None,
    seqlen_k=None,
    window_size=None,
):
    """
    Memory-efficient attention forward pass (xFormers style).

    Args:
        query: (batch, seq_q, num_heads, head_dim)
        key: (batch, seq_k, num_heads, head_dim)
        value: (batch, seq_k, num_heads, head_dim)
        bias: Optional attention bias (batch, num_heads, seq_q, seq_k)
        cu_seqlens_q: Cumulative sequence lengths for query (for variable length)
        cu_seqlens_k: Cumulative sequence lengths for key (for variable length)
        max_seqlen_q: Maximum sequence length for query
        max_seqlen_k: Maximum sequence length for key
        dropout_p: Dropout probability (must be 0.0 currently)
        custom_mask_type: 0=no mask, 1=causal, 2=upper triangular (not supported)
        compute_log_sumexp: Whether to compute and return logsumexp
        scale: Scale factor for attention scores (default: 1/sqrt(head_dim))
        seqlen_k: Per-batch sequence lengths for key
        window_size: Sliding window size (not supported)

    Returns:
        Tuple of (output, logsumexp, philox_seed, philox_offset,
                  max_seqlen_batch_q, max_seqlen_batch_k)
    """
    logger.debug("GEMS _EFFICIENT_ATTENTION_FORWARD")

    # Validate inputs
    assert cu_seqlens_q is None and cu_seqlens_k is None, (
        "Variable length attention (cu_seqlens) is not supported yet"
    )
    assert custom_mask_type in (0, 1), (
        f"custom_mask_type={custom_mask_type} not supported, only 0 (no mask) and 1 (causal) are supported"
    )
    assert window_size is None, "Sliding window attention is not supported yet"

    # Input shape: (batch, seq_len, num_heads, head_dim)
    # Need to transpose to (batch, num_heads, seq_len, head_dim) for the kernel
    batch_size, seq_q, num_heads, head_dim = query.shape
    _, seq_k, _, _ = key.shape

    # Transpose: (batch, seq_len, heads, head_dim) -> (batch, heads, seq_len, head_dim)
    query_t = query.transpose(1, 2).contiguous()
    key_t = key.transpose(1, 2).contiguous()
    value_t = value.transpose(1, 2).contiguous()

    # Handle bias - it should be (batch, num_heads, seq_q, seq_k)
    # The FlagGems kernel applies the mask after converting QK^T to log2 space,
    # so we need to scale the bias by log2(e) for correct behavior
    LOG2E = 1.44269504
    attn_mask = None
    if bias is not None:
        attn_mask = bias * LOG2E

    # Determine if causal mask is needed
    is_causal = custom_mask_type == 1

    # Call the underlying attention implementation
    output_t, M = scaled_dot_product_attention_forward(
        query_t,
        key_t,
        value_t,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=False,
    )

    # Transpose output back: (batch, heads, seq_len, head_dim) -> (batch, seq_len, heads, head_dim)
    output = output_t.transpose(1, 2).contiguous()

    # Prepare logsumexp output
    if compute_log_sumexp:
        # M has shape (batch, num_heads, seq_q)
        # Need to convert from log2 space to natural log space
        # M is stored as log2(sum(exp(x))) so we need to multiply by ln(2) to get natural log
        logsumexp = M * 0.6931471824645996  # ln(2)
    else:
        # Return empty tensor when not computing logsumexp
        logsumexp = torch.empty(
            (batch_size, num_heads, 0),
            device=query.device,
            dtype=torch.float32,
        )

    # Create dummy philox state tensors (we don't support dropout)
    philox_seed = torch.tensor(0, device=query.device, dtype=torch.int64)
    philox_offset = torch.tensor(0, device=query.device, dtype=torch.int64)

    # Return max sequence lengths
    max_seqlen_batch_q = seq_q
    max_seqlen_batch_k = seq_k

    return (
        output,
        logsumexp,
        philox_seed,
        philox_offset,
        max_seqlen_batch_q,
        max_seqlen_batch_k,
    )
