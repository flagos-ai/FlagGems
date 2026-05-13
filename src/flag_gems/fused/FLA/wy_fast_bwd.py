# Copyright (c) 2025 FlagGems. All rights reserved.
# Backward pass for WY representation preparation in chunk_gated_delta_rule.
# ruff: noqa: E501

import torch
import triton
import triton.language as tl

from flag_gems.fused.FLA.index import prepare_chunk_indices
from flag_gems.fused.FLA.triton_ops_helper import exp
from flag_gems.utils import libentry, libtuner


@libentry()
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@libtuner(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def bwd_prepare_wy_repr_kernel(
    k,
    v,
    beta,
    g,
    A_inv,
    dw,
    du,
    dk_out,
    dv_out,
    dbeta_out,
    dg_out,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Backward through WY preparation: computes dk, dv, dbeta, dg from dw, du.

    Accumulates dbeta and dg locally (across K/V blocks) to avoid atomics.
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # Load beta and g for this chunk
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    b_g_exp = exp(b_g)

    # Load A_inv_chunk (BT x BT)
    p_A_inv = tl.make_block_ptr(
        A_inv + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    b_A_inv = tl.load(p_A_inv, boundary_check=(0, 1)).to(tl.float32)
    b_A_inv_T = tl.trans(b_A_inv)

    k_offset = bos * Hg + i_h // (H // Hg)
    v_offset = bos * H + i_h
    w_offset = bos * H + i_h

    # Accumulate dbeta and dg locally across K and V blocks
    b_dbeta_acc = tl.zeros(
        [
            BT,
        ],
        dtype=tl.float32,
    )
    b_dg_acc = tl.zeros(
        [
            BT,
        ],
        dtype=tl.float32,
    )

    # Process K dimension in blocks
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + k_offset * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_dw = tl.make_block_ptr(
            dw + w_offset * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_dw = tl.load(p_dw, boundary_check=(0, 1)).to(tl.float32)

        # Direct gradient: d(k_beta_g) = A_inv^T @ dw
        b_d_k_beta_g = tl.dot(b_A_inv_T, b_dw)

        # dk from k_beta_g: dk += d(k_beta_g) * beta * exp(g)
        b_dk = b_d_k_beta_g * b_beta[:, None] * b_g_exp[:, None]

        # Accumulate dbeta and dg from this K block
        b_dbeta_acc += tl.sum(b_d_k_beta_g * b_k * b_g_exp[:, None], axis=1)
        b_dg_acc += tl.sum(
            b_d_k_beta_g * b_k * b_beta[:, None] * b_g_exp[:, None], axis=1
        )

        # Store dk
        p_dk = tl.make_block_ptr(
            dk_out + k_offset * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    # Process V dimension in blocks
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + v_offset * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_du = tl.make_block_ptr(
            du + v_offset * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_du = tl.load(p_du, boundary_check=(0, 1)).to(tl.float32)

        # Direct gradient: d(v_beta) = A_inv^T @ du
        b_d_v_beta = tl.dot(b_A_inv_T, b_du)

        # dv = d(v_beta) * beta
        b_dv = b_d_v_beta * b_beta[:, None]

        # Accumulate dbeta from this V block
        b_dbeta_acc += tl.sum(b_d_v_beta * b_v, axis=1)

        # Store dv
        p_dv = tl.make_block_ptr(
            dv_out + v_offset * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    # Write accumulated dbeta and dg
    p_dbeta_out = tl.make_block_ptr(
        dbeta_out + bos * H + i_h,
        (T,),
        (H,),
        (i_t * BT,),
        (BT,),
        (0,),
    )
    p_dg_out = tl.make_block_ptr(
        dg_out + bos * H + i_h,
        (T,),
        (H,),
        (i_t * BT,),
        (BT,),
        (0,),
    )
    tl.store(
        p_dbeta_out, b_dbeta_acc.to(p_dbeta_out.dtype.element_ty), boundary_check=(0,)
    )
    tl.store(p_dg_out, b_dg_acc.to(p_dg_out.dtype.element_ty), boundary_check=(0,))


def bwd_prepare_wy_repr(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    A_inv: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward through WY representation preparation.

    Args:
        k, v, beta, g: Saved tensors from forward pass
        A_inv: The inverse matrix from tril_solve (B, T, H, BT)
        dw: Gradient w.r.t. w (B, T, H, K)
        du: Gradient w.r.t. u (B, T, H, V)
        cu_seqlens: Optional cumulative sequence lengths
        chunk_size: Chunk size (BT)

    Returns:
        dk2, dv, dbeta, dg2: Additional gradients to be added to existing ones
    """
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    V = v.shape[-1]
    BT = chunk_size

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)

    dk2 = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dbeta = torch.zeros_like(beta)
    dg2 = torch.zeros_like(g)

    def grid(meta):
        return (NT, B * H)

    bwd_prepare_wy_repr_kernel[grid](
        k=k,
        v=v,
        beta=beta,
        g=g,
        A_inv=A_inv,
        dw=dw,
        du=du,
        dk_out=dk2,
        dv_out=dv,
        dbeta_out=dbeta,
        dg_out=dg2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dk2, dv, dbeta, dg2
