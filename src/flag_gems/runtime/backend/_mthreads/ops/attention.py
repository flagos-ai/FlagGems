import logging

import torch

logger = logging.getLogger("flag_gems.runtime.backend._mthreads.ops.attention")


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """Mthreads scaled_dot_product_attention.

    The shared Triton flash-attention kernel in ``ops/attention.py`` runs ~10x
    slower than the native MUSA fused kernel on mthreads hardware: the autotune
    config space exposed for this backend (a single ``BLOCK_M=BLOCK_N=32,
    stages=1`` config) is far too small, so the kernel never reaches the native
    flash kernel's occupancy. Delegate forward and backward to
    ``torch.nn.functional.scaled_dot_product_attention``, which dispatches to
    the native MUSA fused kernels on both passes. This keeps accuracy identical
    (the same kernels PyTorch itself uses) while restoring competitive
    performance, and avoids the shared Triton backward kernel's contiguity /
    softmax-statistic assumptions that do not hold for the native forward.
    """
    logger.debug("GEMS_MTHREADS SCALED_DOT_PRODUCT_ATTENTION")
    assert dropout_p == 0.0, "Currently only support dropout_p=0.0"
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
