"""
FLA backend implementation of chunk_gated_delta_rule.
Uses FLA's high-level API to match reference implementation, and internal function to get intermediate values.
"""

import torch

from flag_gems.fused.ChunkGatedDeltaRule.chunk_delta_h_fla import (
    chunk_gated_delta_rule_fwd_h,
)
from flag_gems.fused.ChunkGatedDeltaRule.chunk_o_fla import chunk_fwd_o
from flag_gems.fused.ChunkGatedDeltaRule.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fla_fwd,
)
from flag_gems.fused.ChunkGatedDeltaRule.cumsum import chunk_local_cumsum
from flag_gems.fused.ChunkGatedDeltaRule.solve_tril import solve_tril
from flag_gems.fused.ChunkGatedDeltaRule.wy_fast_fla import recompute_w_u_fwd


def chunk_gated_delta_rule_fla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple:
    """
    FLA backend implementation of chunk_gated_delta_rule.

    Uses FLA's high-level API (ChunkGatedDeltaRuleFunction.apply) to match the reference implementation.
    Also calls the internal function to get intermediate values (g_processed, A).

    Args:
        q: [B, T, H, K] query tensor
        k: [B, T, H, K] key tensor (L2 normalized)
        v: [B, T, H, V] value tensor
        g: [B, T, H] gate tensor (in log space)
        beta: [B, T, H] beta tensor
        scale: scaling factor for query
        initial_state: [B, H, K, V] or None
        output_final_state: whether to output final state
        cu_seqlens: cumulative sequence lengths

    Returns:
        o: [B, T, H, V] output tensor
        final_state: [B, H, K, V] or None
    """
    BT = 64
    g = chunk_local_cumsum(g, chunk_size=BT, cu_seqlens=cu_seqlens)
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fla_fwd(
        k=k, beta=beta, g=g, cu_seqlens=cu_seqlens, chunk_size=BT, output_dtype=k.dtype
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=BT,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        chunk_size=BT,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state
