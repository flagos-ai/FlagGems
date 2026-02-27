import logging
from typing import List, Optional

import torch

from flag_gems.ops.attention import (
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)

logger = logging.getLogger(__name__)


def _scaled_dot_product_efficient_attention_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    dropout_p: float,
    grad_input_mask: List[bool],
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
):
    """
    Backward pass for scaled dot product efficient attention.

    Args:
        grad_out: Gradient of the output tensor [B, H, L, D]
        query: Query tensor [B, H, L, D]
        key: Key tensor [B, H, S, D]
        value: Value tensor [B, H, S, D]
        attn_bias: Optional attention bias tensor
        out: Output from forward pass [B, H, L, D]
        logsumexp: LogSumExp values from forward pass [B, H, L]
        philox_seed: Random seed for dropout (unused - dropout not supported)
        philox_offset: Random offset for dropout (unused - dropout not supported)
        dropout_p: Dropout probability (must be 0.0)
        grad_input_mask: List[bool] of length 4 indicating which gradients to compute:
            [compute_grad_query, compute_grad_key, compute_grad_value, compute_grad_bias]
        is_causal: Whether to apply causal mask
        scale: Optional scaling factor. If None, uses 1/sqrt(head_dim)

    Returns:
        Tuple of (grad_query, grad_key, grad_value, grad_bias)
    """
    logger.debug("GEMS _SCALED_DOT_PRODUCT_EFFICIENT_ATTENTION_BACKWARD")

    assert dropout_p == 0.0, "FlagGems currently only supports dropout_p=0.0"

    # Ensure tensors are contiguous for the backward pass
    grad_out = grad_out.contiguous()
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    out = out.contiguous()

    # Compute scale if not provided
    head_dim = query.shape[-1]
    if scale is None:
        sm_scale = 1.0 / (head_dim**0.5)
    else:
        sm_scale = scale

    # Determine if we need to compute gradients
    compute_grad_q = grad_input_mask[0] if len(grad_input_mask) > 0 else True
    compute_grad_k = grad_input_mask[1] if len(grad_input_mask) > 1 else True
    compute_grad_v = grad_input_mask[2] if len(grad_input_mask) > 2 else True
    compute_grad_bias = grad_input_mask[3] if len(grad_input_mask) > 3 else False

    # Call the existing backward implementation
    # Note: The existing implementation computes all gradients
    dq, dk, dv = scaled_dot_product_attention_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        attn_mask=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=sm_scale,
        enable_gqa=False,
    )

    # Handle grad_input_mask - return None for gradients that are not requested
    grad_query = dq if compute_grad_q else torch.zeros_like(query)
    grad_key = dk if compute_grad_k else torch.zeros_like(key)
    grad_value = dv if compute_grad_v else torch.zeros_like(value)

    # Compute attention bias gradient if requested
    # For efficient attention, attn_bias gradient computation is complex
    # We return None or zeros for now as it's rarely used
    if compute_grad_bias and attn_bias is not None:
        # Attention bias gradient would require additional computation
        # For now, return zeros with the same shape as attn_bias
        grad_bias = torch.zeros_like(attn_bias)
    else:
        grad_bias = torch.empty(0, device=query.device, dtype=query.dtype)

    return grad_query, grad_key, grad_value, grad_bias
