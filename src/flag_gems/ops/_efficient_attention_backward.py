import logging
import math

import torch

logger = logging.getLogger(__name__)


def efficient_attention_backward(
    grad_out,
    query,
    key,
    value,
    bias,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    logsumexp,
    dropout_p,
    philox_seed,
    philox_offset,
    custom_mask_type,
    bias_requires_grad,
    *,
    scale=None,
    num_splits_key=None,
    window_size=None,
    shared_storage_dqdkdv=False,
):
    """
    Backward pass for efficient (memory-efficient) attention.

    Computes gradients for query, key, value using standard attention
    backward computation with PyTorch operations.

    Args:
        grad_out: Gradient of the output [batch, seq_q, num_heads, head_dim]
        query: Query tensor [batch, seq_q, num_heads, head_dim]
        key: Key tensor [batch, seq_k, num_heads_k, head_dim]
        value: Value tensor [batch, seq_k, num_heads_k, head_dim]
        bias: Optional attention bias tensor
        out: Forward pass output tensor
        cu_seqlens_q: Cumulative sequence lengths for queries
        cu_seqlens_k: Cumulative sequence lengths for keys
        max_seqlen_q: Maximum query sequence length
        max_seqlen_k: Maximum key sequence length
        logsumexp: Log-sum-exp values from forward pass [batch, num_heads, seq_q]
        dropout_p: Dropout probability (must be 0.0)
        philox_seed: Random seed for dropout (unused if dropout_p=0)
        philox_offset: Random offset for dropout (unused if dropout_p=0)
        custom_mask_type: Mask type (0=no mask, 1=causal, 2=causal with padding)
        bias_requires_grad: Whether to compute gradient for bias
        scale: Attention scale (default: 1/sqrt(head_dim))
        num_splits_key: Number of splits for key (unused)
        window_size: Window size for local attention (unused)
        shared_storage_dqdkdv: Whether to use shared storage (unused)

    Returns:
        Tuple of (grad_query, grad_key, grad_value, grad_bias)
    """
    logger.debug("GEMS EFFICIENT ATTENTION BACKWARD")

    # Currently only support dense attention (no variable-length sequences)
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        raise NotImplementedError(
            "Variable-length sequences not supported in _efficient_attention_backward"
        )

    # Currently only support dropout_p=0
    if dropout_p != 0.0:
        raise NotImplementedError(
            "Non-zero dropout not supported in _efficient_attention_backward"
        )

    # Input shapes: [batch, seq, num_heads, head_dim]
    batch_size, seq_q, num_heads, head_dim = query.shape
    _, seq_k, num_heads_k, _ = key.shape

    # Calculate scale
    if scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    else:
        sm_scale = scale

    # Determine if causal mask is used
    is_causal = custom_mask_type == 1 or custom_mask_type == 2

    # Transpose to [batch, num_heads, seq, head_dim] for computation
    # Use view/permute to avoid copy where possible
    q = query.transpose(1, 2)  # [batch, num_heads, seq_q, head_dim]
    k = key.transpose(1, 2)    # [batch, num_heads_k, seq_k, head_dim]
    v = value.transpose(1, 2)  # [batch, num_heads_k, seq_k, head_dim]
    do = grad_out.transpose(1, 2)  # [batch, num_heads, seq_q, head_dim]

    # Handle grouped-query attention (GQA)
    # Expand k and v to match q's number of heads
    group_size = num_heads // num_heads_k
    if group_size > 1:
        k = k.repeat_interleave(group_size, dim=1)  # [batch, num_heads, seq_k, head_dim]
        v = v.repeat_interleave(group_size, dim=1)  # [batch, num_heads, seq_k, head_dim]

    # Compute attention scores: [batch, num_heads, seq_q, seq_k]
    # scores = Q @ K^T * scale
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    # Apply causal mask if needed
    if is_causal:
        causal_mask = torch.ones(seq_q, seq_k, dtype=torch.bool, device=query.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        scores = scores.masked_fill(causal_mask, float('-inf'))

    # Apply bias if provided
    if bias is not None:
        scores = scores + bias

    # Compute softmax probabilities: [batch, num_heads, seq_q, seq_k]
    # We could use logsumexp for numerical stability, but PyTorch softmax is stable
    probs = torch.softmax(scores, dim=-1)

    # Handle NaN from all-masked rows (for causal attention)
    if is_causal:
        probs = torch.nan_to_num(probs, nan=0.0)

    # Compute gradients
    # dV = P^T @ dO
    # [batch, num_heads, seq_k, seq_q] @ [batch, num_heads, seq_q, head_dim]
    # = [batch, num_heads, seq_k, head_dim]
    dv = torch.matmul(probs.transpose(-2, -1), do)

    # dP = dO @ V^T
    # [batch, num_heads, seq_q, head_dim] @ [batch, num_heads, head_dim, seq_k]
    # = [batch, num_heads, seq_q, seq_k]
    dp = torch.matmul(do, v.transpose(-2, -1))

    # Softmax backward: dS = P * (dP - sum(P * dP, dim=-1, keepdim=True))
    # This is the Jacobian-vector product for softmax
    sum_dp_p = torch.sum(dp * probs, dim=-1, keepdim=True)
    ds = probs * (dp - sum_dp_p)

    # Apply causal mask gradient (masked positions have zero gradient)
    if is_causal:
        ds = ds.masked_fill(causal_mask, 0.0)

    # Scale the score gradients
    ds = ds * sm_scale

    # dQ = dS @ K
    # [batch, num_heads, seq_q, seq_k] @ [batch, num_heads, seq_k, head_dim]
    # = [batch, num_heads, seq_q, head_dim]
    dq = torch.matmul(ds, k)

    # dK = dS^T @ Q
    # [batch, num_heads, seq_k, seq_q] @ [batch, num_heads, seq_q, head_dim]
    # = [batch, num_heads, seq_k, head_dim]
    dk = torch.matmul(ds.transpose(-2, -1), q)

    # Handle GQA: sum gradients across grouped heads
    if group_size > 1:
        # Reshape to [batch, num_heads_k, group_size, seq_k, head_dim]
        dk = dk.view(batch_size, num_heads_k, group_size, seq_k, head_dim)
        dv = dv.view(batch_size, num_heads_k, group_size, seq_k, head_dim)
        # Sum across group dimension
        dk = dk.sum(dim=2)  # [batch, num_heads_k, seq_k, head_dim]
        dv = dv.sum(dim=2)  # [batch, num_heads_k, seq_k, head_dim]

    # Transpose back to [batch, seq, num_heads, head_dim]
    grad_query = dq.transpose(1, 2).contiguous()
    grad_key = dk.transpose(1, 2).contiguous()
    grad_value = dv.transpose(1, 2).contiguous()

    # Compute bias gradient if needed
    if bias_requires_grad and bias is not None:
        # dBias = sum over batch of dS (before scaling)
        # The bias is added to scores, so its gradient is ds / sm_scale
        grad_bias = (ds / sm_scale).sum(dim=0) if bias.dim() == 3 else ds / sm_scale
    else:
        grad_bias = torch.empty(0, device=query.device, dtype=query.dtype)

    return grad_query, grad_key, grad_value, grad_bias
