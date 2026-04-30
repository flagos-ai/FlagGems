"""
GagedDeltaNet backend implementation of chunk_gated_delta_rule.
Uses the exact GatedDeltaNet implementation for perfect numerical compatibility.
"""

import torch

# Import the exact GatedDeltaNet implementation
# from flag_gems.fused.FLA.chunk_gated_delta_reference import chunk_gated_delta_rule as gated_delta_rule_func
from flag_gems.fused.ChunkGatedDeltaRule.chunk_optimized import (
    chunk_gated_delta_rule as gated_delta_rule_func,
)


def chunk_gated_delta_rule_gateddnet(
    q: torch.Tensor,  # [B, H, T, K]
    k: torch.Tensor,  # [B, H, T, K]
    v: torch.Tensor,  # [B, H, T, V]
    g: torch.Tensor,  # [B, H, T]
    beta: torch.Tensor,  # [B, H, T]
    scale: float,
    initial_state: torch.Tensor,  # [B, H, K, V] or None
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    BT: int = 64,
    dtype: torch.dtype = torch.float32,
) -> tuple:
    """
    GagedDeltaNet backend implementation of chunk_gated_delta_rule.
    Uses the exact GatedDeltaNet implementation for perfect numerical compatibility.

    Args:
        q: [B, H, T, K] query tensor (head-first layout)
        k: [B, H, T, K] key tensor (head-first layout, L2 normalized)
        v: [B, H, T, V] value tensor (head-first layout)
        g: [B, H, T] gate tensor (head-first layout, in log space)
        beta: [B, H, T] beta tensor (head-first layout)
        scale: scaling factor for query (not used, for compatibility)
        initial_state: [B, H, K, V] or None
        output_final_state: whether to output final state
        cu_seqlens: cumulative sequence lengths
        BT: chunk size
        dtype: data type of input tensors (for float32-specific optimizations)

    Returns:
        g_processed: [B, H, T] processed gate tensor
        o: [B, H, T, V] output tensor
        A: None (not used in GagedDeltaNet)
        final_state: [B, H, K, V] or None
        w: None (not used in GagedDeltaNet)
        h: None (not used in GagedDeltaNet)
        v_new: None (not used in GagedDeltaNet)
    """
    if cu_seqlens is not None:
        raise NotImplementedError(
            "cu_seqlens is not supported in GatedDeltaNet backend"
        )

    # Call the exact GatedDeltaNet implementation
    # Parameters are in the order: (q, k, v, beta, g, BT, ...)
    o, final_state = gated_delta_rule_func(
        q,
        k,
        v,
        beta,
        g,
        BT=BT,
        initial_state=initial_state,
        output_final_state=output_final_state,
        dtype=dtype,
    )

    # Return 7-tuple to match FlagGems interface (all in head-first format)
    return g, o, None, final_state, None, None, None
