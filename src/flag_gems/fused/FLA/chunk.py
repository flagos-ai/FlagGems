# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import logging

import torch
import torch.nn.functional as F

from flag_gems.fused.FLA.chunk_bwd import chunk_bwd_dhu, chunk_bwd_dqkw, fwd_prepare_du
from flag_gems.fused.FLA.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from flag_gems.fused.FLA.chunk_o import chunk_fwd_o
from flag_gems.fused.FLA.fused_cumsum_kkt_solve_tril import (
    chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril,
)
from flag_gems.fused.FLA.utils import SUPPRESS_LEVEL
from flag_gems.fused.FLA.wy_fast import recompute_w_u_fwd
from flag_gems.fused.FLA.wy_fast_bwd import bwd_prepare_wy_repr

logger = logging.getLogger(__name__)


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
):
    logger.debug("GEMS CHUNK GATED DELTA RULE FWD")
    g, A = chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
        g=g, k=k, beta=beta, cu_seqlens=cu_seqlens, chunk_size=64, output_dtype=k.dtype
    )
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
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """Autograd Function for chunk_gated_delta_rule with full forward+backward.

    This wraps the chunked gated delta rule computation used in Qwen3-Next architecture.
    It supports both training (with gradient computation) and inference.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        # Step 1: cumsum + KKT + tril_solve
        # Use float32 for internal computation for numerical stability
        q_f32 = q.float()
        k_f32 = k.float()
        v_f32 = v.float()
        beta_f32 = beta.float()
        g_f32 = g.float()
        g_cumsum, A_inv = chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
            g=g_f32,
            k=k_f32,
            beta=beta_f32,
            cu_seqlens=cu_seqlens,
            chunk_size=64,
            output_dtype=torch.float32,
        )

        # Step 2: WY representation
        w, u = recompute_w_u_fwd(
            k=k_f32,
            v=v_f32,
            beta=beta_f32,
            A=A_inv,
            g_cumsum=g_cumsum,
            cu_seqlens=cu_seqlens,
        )

        # Step 3: Hidden state recurrence
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=k_f32,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

        # Step 4: Output computation
        o = chunk_fwd_o(
            q=q_f32,
            k=k_f32,
            v=v_new,
            h=h,
            g=g_cumsum,
            scale=scale,
            cu_seqlens=cu_seqlens,
        )

        # Save for backward: include w, h, v_new to avoid recomputation.
        # Memory cost (float32, T=1024): w=2MB, h=2MB, v_new=2MB → 6MB total.
        ctx.save_for_backward(
            q_f32,
            k_f32,
            v_f32,
            beta_f32,
            g_f32,
            A_inv,
            g_cumsum,
            initial_state,
            w,  # saved to skip recompute_w_u_fwd in backward
            h,  # saved to skip chunk_fwd_h recomputation
            v_new,  # saved to skip chunk_fwd_h recomputation
        )
        ctx.cu_seqlens = cu_seqlens
        ctx.output_final_state = output_final_state
        ctx.original_dtype = q.dtype

        return o.to(ctx.original_dtype), final_state

    @staticmethod
    def backward(ctx, do: torch.Tensor, d_final_state: torch.Tensor | None = None):
        (
            q,
            k,
            v,
            beta,
            g,
            A_inv,
            g_cumsum,
            initial_state,
            w,
            h,
            v_new,
        ) = ctx.saved_tensors
        cu_seqlens = ctx.cu_seqlens

        # Cast do to float32 for numerical stability
        do = do.float()

        # Step 1: fwd_prepare_du - initial dv from do
        du = fwd_prepare_du(
            q=q,
            k=k,
            g=g_cumsum,
            do=do,
            cu_seqlens=cu_seqlens,
        )

        # Step 2: chunk_bwd_dhu - backward through hidden state
        dh, du = chunk_bwd_dhu(
            q=q,
            k=k,
            w=w,
            g=g_cumsum,
            do=do,
            dv=du,
            cu_seqlens=cu_seqlens,
        )

        # Get shapes from saved tensors
        B, T, Hg_val, K_val = q.shape
        H_val = beta.shape[-1]

        # Step 3: chunk_bwd_dqkw - compute dq, dk, dw, dg
        dq, dk, dw, dg_flat = chunk_bwd_dqkw(
            q=q,
            k=k,
            v_new=v_new,
            w=w,
            g=g_cumsum,
            h=h,
            do=do,
            dh=dh,
            dv=du,
            cu_seqlens=cu_seqlens,
        )
        # dg from dqkw has shape (B*H, T), reshape to (B, T, H)
        dg = dg_flat.reshape(B, H_val, T).transpose(1, 2).contiguous()

        # Step 4: bwd_prepare_wy_repr - invert WY representation
        dk2, dv, dbeta, dg2 = bwd_prepare_wy_repr(
            k=k,
            v=v,
            beta=beta,
            g=g_cumsum,
            A_inv=A_inv,
            dw=dw,
            du=du,
            cu_seqlens=cu_seqlens,
        )

        dk = dk + dk2
        dg = dg + dg2

        # Reverse the cumsum on gate gradients
        # Forward does cumsum within each chunk: g_cumsum[t] = sum_{i=0}^{t} g[i] within chunk
        # Backward: reverse_cumsum(dg_cumsum) to get dg_original
        dg = dg.float()
        B, T, H = dg.shape
        BT = 64
        # Reshape: (B, T, H) -> (B, NT, BT, H)
        dg = dg.reshape(B, -1, BT, H)
        # Reverse cumsum along the BT dimension
        cumsum_dg = dg.cumsum(-2)
        rev_cumsum_dg = cumsum_dg[..., -1, None, :] - cumsum_dg
        dg = rev_cumsum_dg + dg
        # Reshape back: (B, NT, BT, H) -> (B, T, H)
        dg = dg.reshape(B, T, H)

        orig_dtype = ctx.original_dtype
        return (
            dq.to(orig_dtype),
            dk.to(orig_dtype),
            dv.to(orig_dtype),
            dbeta.to(orig_dtype),
            dg.to(orig_dtype),
            None,  # scale
            None,  # initial_state
            None,  # output_final_state
            None,  # cu_seqlens
        )


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    BT: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked Gated Delta Rule operator for Qwen3-Next architecture.

    Implements the chunked parallel form of the gated delta rule using
    the WY representation for efficient linear attention.

    Args:
        q: Query tensor of shape (B, T, Hg, K) where Hg = H or H * tp_size
        k: Key tensor of shape (B, T, Hg, K)
        v: Value tensor of shape (B, T, H, V)
        beta: Gate tensor of shape (B, T, H) in range [0, 1]
        g: Log decay tensor of shape (B, T, H), typically negative values
        scale: Optional scaling factor for QK attention (default: K^{-0.5})
        initial_state: Optional initial hidden state of shape (B, H, K, V)
        output_final_state: Whether to return final hidden state
        cu_seqlens: Optional cumulative sequence lengths for varlen support
        BT: Chunk size (default: 64)

    Returns:
        o: Output tensor of shape (B, T, H, V)
        final_state: Optional final hidden state of shape (B, H, K, V)
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    assert q.dtype == k.dtype == v.dtype, "q, k, v must have the same dtype"
    # q shape: (B, T, Hg, K), so sequence length is at dim=1
    L = q.shape[1]

    # Pad sequence length to multiple of BT
    if L % BT != 0:
        pad_len = BT - L % BT
        # q, k shape: (B, T, Hg, K), pad the T dimension (dim=1)
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        # v shape: (B, T, H, V), pad the T dimension (dim=1)
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        # beta, g shape: (B, T, H), pad the T dimension (dim=1)
        beta = F.pad(beta, (0, 0, 0, pad_len))
        g = F.pad(g, (0, 0, 0, pad_len))

    if initial_state is not None:
        initial_state = initial_state.detach()

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        beta,
        g,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
    )

    # Unpad output
    if L % BT != 0:
        o = o[:, :L]

    return o, final_state
