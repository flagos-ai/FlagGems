# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import torch

# Import backend implementations
from flag_gems.fused.ChunkGatedDeltaRule.chunk_fla import chunk_gated_delta_rule_fla
from flag_gems.fused.ChunkGatedDeltaRule.chunk_gateddnet import (
    chunk_gated_delta_rule_gateddnet,
)


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    backend: str = "gateddnet",  # 'fla' or 'gateddnet'
    BT: int = 64,
):
    """
    Chunk gated delta rule forward pass with backend selection.

    Args:
        q: Query tensor
           - For 'fla' backend: [B, T, H, K] (time-first layout)
           - For 'gateddnet' backend: [B, H, T, K] (head-first layout)
        k: Key tensor (L2 normalized)
           - For 'fla' backend: [B, T, H, K] (time-first layout)
           - For 'gateddnet' backend: [B, H, T, K] (head-first layout)
        v: Value tensor
           - For 'fla' backend: [B, T, H, V] (time-first layout)
           - For 'gateddnet' backend: [B, H, T, V] (head-first layout)
        g: Gate tensor (in log space)
           - For 'fla' backend: [B, T, H] (time-first layout)
           - For 'gateddnet' backend: [B, H, T] (head-first layout)
        beta: Beta tensor
           - For 'fla' backend: [B, T, H] (time-first layout)
           - For 'gateddnet' backend: [B, H, T] (head-first layout)
        scale: Scaling factor for query
        initial_state: [B, H, K, V] or None (head-first layout)
        output_final_state: Whether to output final state
        cu_seqlens: Cumulative sequence lengths
        backend: 'fla' or 'gateddnet' (default: 'gateddnet')
            - 'fla': Uses FLA reference implementation
            - 'gateddnet': Uses FlagGems optimized implementation (GagedDeltaNet compatible)
        BT: Chunk size for processing (default: 64)

    Returns:
        g_processed: [B, H, T] processed gate tensor
        o: [B, H, T, V] output tensor
        A: None
        final_state: [B, H, K, V] or None
        w: None
        h: None
        v_new: None

    Note:
        Different backends expect different input layouts to avoid stride differences
        that cause Triton autotuner to select different kernel configurations.
        Always create tensors in the format expected by the chosen backend.
    """
    if backend == "fla":
        # Call FLA backend with time-first layout
        # Assume input is already in time-first format [B, T, H, K/V]
        o_out, final_state = chunk_gated_delta_rule_fla(
            q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens
        )
        # Output is in time-first format [B, T, H, V]
        return None, o_out, None, final_state, None, None, None

    elif backend == "gateddnet":
        # Call GagedDeltaNet backend with head-first layout
        g_out, o_out, _, final_state, _, _, _ = chunk_gated_delta_rule_gateddnet(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            BT=BT,
            dtype=q.dtype,
        )

        return g_out, o_out, None, final_state, None, None, None
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'fla' or 'gateddnet'.")
