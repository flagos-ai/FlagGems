# Copyright (c) 2025 FlagGems. All rights reserved.
# Backward kernels for chunk_gated_delta_rule operator.
# ruff: noqa: E501

import torch
import triton
import triton.language as tl

from flag_gems.fused.FLA.index import prepare_chunk_indices
from flag_gems.fused.FLA.triton_ops_helper import exp
from flag_gems.utils import libentry, libtuner

# ======================== fwd_prepare_du ========================
# Computes initial dv gradient using causal attention matrix A = (k @ q^T) * exp(g_j - g_i)


@libentry()
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def fwd_prepare_du_kernel(
    q,
    k,
    g,
    do,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
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

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + (bos * Hg + i_h // (H // Hg)) * K,
            (K, T),
            (1, Hg * K),
            (i_k * BK, i_t * BT),
            (BK, BT),
            (0, 1),
        )
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_k.dtype)
        b_A += tl.dot(b_k, b_q)

    if USE_G:
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,)
        )
        b_g = tl.load(p_g, boundary_check=(0,))
        b_A = b_A * exp(b_g[None, :] - b_g[:, None])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    b_A = tl.where((o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t), b_A, 0).to(
        do.dtype.element_ty
    )

    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(
            do + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_dv = tl.make_block_ptr(
            dv + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.dot(b_A, b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def fwd_prepare_du(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, Hg, K, V = *k.shape, do.shape[-1]
    H = do.shape[-2]
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    dv = torch.empty_like(do)

    def grid(meta):
        return (NT, B * H)

    fwd_prepare_du_kernel[grid](
        q=q,
        k=k,
        g=g,
        do=do,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=K**-0.5,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return dv


# ======================== chunk_bwd_dhu ========================
# Backward through hidden state recurrence


@libentry()
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=["H", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_bwd_kernel_dhu(
    q,
    k,
    w,
    g,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK=64, BV]
    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # offset calculation
    dh_offset = (boh * H + i_h).to(tl.int64) * K * V
    do_offset = (bos * H + i_h) * V
    v_offset = (bos * H + i_h) * V
    q_offset = (bos * Hg + i_h // (H // Hg)) * K
    k_offset = (bos * Hg + i_h // (H // Hg)) * K
    w_offset = (bos * H + i_h) * K
    stride_h = H * K * V
    stride_v = H * V
    stride_k = Hg * K
    stride_w = H * K

    for i_t in range(NT - 1, -1, -1):
        # store dh for this chunk
        p_dh1 = tl.make_block_ptr(
            dh + dh_offset + i_t * stride_h,
            (K, V),
            (V, 1),
            (0, i_v * BV),
            (64, BV),
            (1, 0),
        )
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh2 = tl.make_block_ptr(
                dh + dh_offset + i_t * stride_h,
                (K, V),
                (V, 1),
                (64, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh3 = tl.make_block_ptr(
                dh + dh_offset + i_t * stride_h,
                (K, V),
                (V, 1),
                (128, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh4 = tl.make_block_ptr(
                dh + dh_offset + i_t * stride_h,
                (K, V),
                (V, 1),
                (192, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        b_dh_tmp1 = tl.zeros([64, BV], dtype=tl.float32)
        b_dh_tmp2 = tl.zeros([64, BV], dtype=tl.float32) if K > 64 else None
        b_dh_tmp3 = tl.zeros([64, BV], dtype=tl.float32) if K > 128 else None
        b_dh_tmp4 = tl.zeros([64, BV], dtype=tl.float32) if K > 192 else None

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + bos * H + last_idx * H + i_h)

        p_w1 = tl.make_block_ptr(
            w + w_offset, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        p_q1 = tl.make_block_ptr(
            q + q_offset, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        p_dv = tl.make_block_ptr(
            dv + v_offset, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_do = tl.make_block_ptr(
            do + do_offset,
            (T, V),
            (stride_v, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )

        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g_chunk = tl.load(p_g, boundary_check=(0,))

        # q * scale * exp(g)
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_do_chunk = tl.load(p_do, boundary_check=(0, 1))
        b_q1 = (b_q1 * scale * exp(b_g_chunk)[None, :]).to(b_q1.dtype)
        b_dh_tmp1 += tl.dot(b_q1, b_do_chunk.to(b_q1.dtype))
        if K > 64:
            p_q2 = tl.make_block_ptr(
                q + q_offset, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_q2 = tl.load(p_q2, boundary_check=(0, 1))
            b_q2 = (b_q2 * scale * exp(b_g_chunk)[None, :]).to(b_q2.dtype)
            b_dh_tmp2 += tl.dot(b_q2, b_do_chunk.to(b_q2.dtype))
        if K > 128:
            p_q3 = tl.make_block_ptr(
                q + q_offset, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_q3 = tl.load(p_q3, boundary_check=(0, 1))
            b_q3 = (b_q3 * scale * exp(b_g_chunk)[None, :]).to(b_q3.dtype)
            b_dh_tmp3 += tl.dot(b_q3, b_do_chunk.to(b_q3.dtype))
        if K > 192:
            p_q4 = tl.make_block_ptr(
                q + q_offset, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_q4 = tl.load(p_q4, boundary_check=(0, 1))
            b_q4 = (b_q4 * scale * exp(b_g_chunk)[None, :]).to(b_q4.dtype)
            b_dh_tmp4 += tl.dot(b_q4, b_do_chunk.to(b_q4.dtype))

        # k * exp(g_last - g)
        p_k1 = tl.make_block_ptr(
            k + k_offset, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_k1 = (b_k1 * exp(b_g_last - b_g_chunk)[:, None]).to(b_k1.dtype)

        # w * exp(g)
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w1 = (b_w1 * exp(b_g_chunk)[:, None]).to(b_w1.dtype)

        b_dv_chunk = tl.load(p_dv, boundary_check=(0, 1))
        b_dv_chunk += tl.dot(b_k1, b_dh1.to(b_k1.dtype))
        if K > 64:
            p_k2 = tl.make_block_ptr(
                k + k_offset, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_k2 = tl.load(p_k2, boundary_check=(0, 1))
            b_k2 = (b_k2 * exp(b_g_last - b_g_chunk)[:, None]).to(b_k2.dtype)
            b_dv_chunk += tl.dot(b_k2, b_dh2.to(b_k2.dtype))
        if K > 128:
            p_k3 = tl.make_block_ptr(
                k + k_offset, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_k3 = tl.load(p_k3, boundary_check=(0, 1))
            b_k3 = (b_k3 * exp(b_g_last - b_g_chunk)[:, None]).to(b_k3.dtype)
            b_dv_chunk += tl.dot(b_k3, b_dh3.to(b_k3.dtype))
        if K > 192:
            p_k4 = tl.make_block_ptr(
                k + k_offset, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_k4 = tl.load(p_k4, boundary_check=(0, 1))
            b_k4 = (b_k4 * exp(b_g_last - b_g_chunk)[:, None]).to(b_k4.dtype)
            b_dv_chunk += tl.dot(b_k4, b_dh4.to(b_k4.dtype))

        p_dv2 = tl.make_block_ptr(
            dv2 + v_offset,
            (T, V),
            (stride_v, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        tl.store(p_dv2, b_dv_chunk.to(p_dv2.dtype.element_ty), boundary_check=(0, 1))

        b_dh_tmp1 -= tl.dot(b_w1, b_dv_chunk.to(b_w1.dtype))
        if K > 64:
            p_w2 = tl.make_block_ptr(
                w + w_offset, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w2 = tl.load(p_w2, boundary_check=(0, 1))
            b_w2 = (b_w2 * exp(b_g_chunk)[:, None]).to(b_w2.dtype)
            b_dh_tmp2 -= tl.dot(b_w2, b_dv_chunk.to(b_w2.dtype))
        if K > 128:
            p_w3 = tl.make_block_ptr(
                w + w_offset, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w3 = tl.load(p_w3, boundary_check=(0, 1))
            b_w3 = (b_w3 * exp(b_g_chunk)[:, None]).to(b_w3.dtype)
            b_dh_tmp3 -= tl.dot(b_w3, b_dv_chunk.to(b_w3.dtype))
        if K > 192:
            p_w4 = tl.make_block_ptr(
                w + w_offset, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w4 = tl.load(p_w4, boundary_check=(0, 1))
            b_w4 = (b_w4 * exp(b_g_chunk)[:, None]).to(b_w4.dtype)
            b_dh_tmp4 -= tl.dot(b_w4, b_dv_chunk.to(b_w4.dtype))

        b_g_last_exp = exp(b_g_last)
        b_dh1 = b_dh1 * b_g_last_exp + b_dh_tmp1
        if K > 64:
            b_dh2 = b_dh2 * b_g_last_exp + b_dh_tmp2
        if K > 128:
            b_dh3 = b_dh3 * b_g_last_exp + b_dh_tmp3
        if K > 192:
            b_dh4 = b_dh4 * b_g_last_exp + b_dh_tmp4


def chunk_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor | None,
    do: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, do.shape[-1]
    H = do.shape[-2]
    BT = chunk_size
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        from flag_gems.fused.FLA.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )

        chunk_indices_tmp = prepare_chunk_indices(cu_seqlens, BT)
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices_tmp),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    dh = q.new_empty(B, NT, H, K, V, dtype=torch.float32)
    dv2 = torch.empty_like(dv)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_bwd_kernel_dhu[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=K**-0.5,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return dh, dv2


# ======================== chunk_bwd_dqkw ========================
# Computes dq, dk, dw, dg


@libentry()
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_bwd_kernel_dqkw(
    q,
    k,
    v,
    w,
    g,
    h,
    do,
    dh,
    dv,
    dq,
    dk,
    dw,
    dg,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dg_last = tl.zeros(
        [
            1,
        ],
        dtype=tl.float32,
    )
    b_dg = tl.zeros(
        [
            BT,
        ],
        dtype=tl.float32,
    )

    last_idx = min((i_t + 1) * BT, T) - 1
    b_g_last = tl.load(g + bos * H + last_idx * H + i_h)

    h_base = h + ((i_b * NT + i_t) * H + i_h).to(tl.int64) * K * V
    q_base = q + (bos * Hg + i_h // (H // Hg)) * K
    k_base = k + (bos * Hg + i_h // (H // Hg)) * K
    v_base = v + (bos * H + i_h) * V
    do_base = do + (bos * H + i_h) * V
    dh_base = dh + ((i_b * NT + i_t) * H + i_h).to(tl.int64) * K * V

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_h = tl.make_block_ptr(
            h_base, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1)
        )
        p_do = tl.make_block_ptr(
            do_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_dh = tl.make_block_ptr(
            dh_base, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1)
        )
        p_dv_in = tl.make_block_ptr(
            dv + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dv_in = tl.load(p_dv_in, boundary_check=(0, 1))
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
        b_dw += tl.dot(b_dv_in, b_h.to(b_dv_in.dtype))

    b_dg_last *= exp(b_g_last)

    p_q = tl.make_block_ptr(
        q_base, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k_base, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    if USE_G:
        p_g_chunk = tl.make_block_ptr(
            g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,)
        )
        b_g = tl.load(p_g_chunk, boundary_check=(0,))
        b_g_exp_qw = exp(b_g)
        b_dq = b_dq * b_g_exp_qw[:, None] * scale
        b_dg += tl.sum(b_dq * b_q, axis=1)
        b_dk = b_dk * exp(b_g_last - b_g)[:, None]
        b_dg -= tl.sum(b_dk * b_k, axis=1)
        b_dg_last += tl.sum(b_dk * b_k)
        b_ds = tl.where(
            o_i[:, None] >= o_i[None, :],
            b_ds * scale * exp(b_g[:, None] - b_g[None, :]),
            0,
        ).to(b_q.dtype)
        b_dg_mask = tl.dot(b_q, tl.trans(b_k)) * b_ds
        b_dg += tl.sum(b_dg_mask, axis=1)
        b_dg -= tl.sum(b_dg_mask, axis=0)
    else:
        b_dq = b_dq * scale
        b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds * scale, 0).to(b_q.dtype)

    b_dq += tl.dot(b_ds, b_k)
    b_dk += tl.trans(tl.dot(tl.trans(b_q), b_ds))

    p_dq = tl.make_block_ptr(
        dq + (bos * Hg + i_h // (H // Hg)) * K,
        (T, K),
        (Hg * K, 1),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + (bos * Hg + i_h // (H // Hg)) * K,
        (T, K),
        (Hg * K, 1),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )
    p_dw = tl.make_block_ptr(
        dw + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))

    # store dg partial results
    dg_idx = i_bh + i_k * B * H
    tl.store(dg + dg_idx * T + i_t * BT + tl.arange(0, BT), b_dg)
    # accumulate dg_last
    b_dg_last_prev = tl.load(dg + dg_idx * T + i_t * BT + BT - 1)
    b_dg_last += b_dg_last_prev
    tl.store(dg + dg_idx * T + i_t * BT + BT - 1 + tl.arange(0, 1), b_dg_last)


def chunk_bwd_dqkw(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor | None,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *q.shape, v_new.shape[-1]
    H = v_new.shape[-2]
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NK = triton.cdiv(K, 64)

    dq = torch.empty_like(q)
    dk = torch.empty_like(q)  # k has same shape as q
    dw = torch.empty(B, T, H, K, device=k.device, dtype=k.dtype)
    # dg is accessed as (NK, B*H, T) in the kernel; use this flat layout
    dg = torch.zeros(NK, B * H, T, device=g.device, dtype=torch.float32)

    def grid(meta):
        return (NK, NT, B * H)

    chunk_gated_delta_rule_bwd_kernel_dqkw[grid](
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        do=do,
        dh=dh,
        dv=dv,
        dq=dq,
        dk=dk,
        dw=dw,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=K**-0.5,
        T=T,
        B=B,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    dg = dg.sum(0)
    return dq, dk, dw, dg
