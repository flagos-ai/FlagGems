# ============================================================================
# [c228] scaled_dot_product_flash_attention_backward via the XPU launch-table
# binding (dense BSHD inputs). Same carrier route as the sdpa backward
# (attention.py: sdnn_fa_bwd_route); this variant carries philox fields that
# the binding contract does not use (dropout_p=0 only).
# ============================================================================

import logging

import torch

from .attention import sdnn_fa_bwd_route
from .contiguous import contiguous
from .to import to_copy

logger = logging.getLogger(__name__)

__all__ = ["scaled_dot_product_flash_attention_backward"]


def scaled_dot_product_flash_attention_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    *,
    scale=None,
):
    logger.debug("GEMS_KUNLUNXIN SCALED_DOT_PRODUCT_FLASH_ATTENTION_BACKWARD")
    if cum_seq_q is not None or cum_seq_k is not None:
        raise NotImplementedError(
            "kunlunxin scaled_dot_product_flash_attention_backward supports dense inputs only"
        )
    if dropout_p and float(dropout_p) > 0.0:
        raise NotImplementedError(
            "kunlunxin scaled_dot_product_flash_attention_backward requires dropout_p=0"
        )
    head_dim = query.shape[-1]
    if scale is None:
        sm_scale = 1.0 / (head_dim**0.5)
    else:
        sm_scale = scale
    # The upstream logsumexp is a non-contiguous [B, H, S] view of a
    # [B, S, H] ordered buffer; the bound kernel reads it as dense [B, H, S].
    lse = contiguous(logsumexp[:, :, : query.shape[1]])
    if lse.dtype != torch.float32:
        lse = to_copy(lse, dtype=torch.float32)
    dq, dk, dv = sdnn_fa_bwd_route(
        contiguous(grad_out),
        contiguous(query),
        contiguous(key),
        contiguous(value),
        contiguous(out),
        lse,
        bool(is_causal),
        sm_scale,
    )
    return dq, dk, dv
