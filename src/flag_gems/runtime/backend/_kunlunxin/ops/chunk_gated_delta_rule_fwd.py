# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _chunk_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,  # sequence length (runtime), used only when cu_seqlens is None
    stride_q_t,
    stride_q_h,
    stride_q_k,
    stride_k_t,
    stride_k_h,
    stride_k_k,
    stride_v_t,
    stride_v_hv,
    stride_v_v,
    stride_o_t,
    stride_o_hv,
    stride_o_v,
    stride_g_t,
    stride_g_hv,
    stride_beta_t,
    stride_beta_hv,
    stride_cu,
    H: tl.constexpr,  # number of q/k heads (Hg)
    HV: tl.constexpr,  # number of v heads
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    H0_STRIDE_S: tl.constexpr,
    H0_STRIDE_HV: tl.constexpr,
    HT_STRIDE_S: tl.constexpr,
    HT_STRIDE_HV: tl.constexpr,
    USE_CU: tl.constexpr,  # cu_seqlens is not None
    STORE_FINAL: tl.constexpr,  # output_final_state
):
    i_seq, i_hv, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if USE_CU:
        t0 = tl.load(cu_seqlens + i_seq * stride_cu).to(tl.int64)
        t1 = tl.load(cu_seqlens + (i_seq + 1) * stride_cu).to(tl.int64)
    else:
        t0 = i_seq.to(tl.int64) * T
        t1 = t0 + T

    # q/k head for this value-head group (grouped heads: H divides HV)
    i_h = i_hv // (HV // H)
    offs = tl.arange(0, BK)

    p_q = q + t0 * stride_q_t + i_h * stride_q_h + offs * stride_q_k
    p_k = k + t0 * stride_k_t + i_h * stride_k_h + offs * stride_k_k
    p_v = v + t0 * stride_v_t + i_hv * stride_v_hv + i_v * stride_v_v
    p_g = g + t0 * stride_g_t + i_hv * stride_g_hv
    p_beta = beta + t0 * stride_beta_t + i_hv * stride_beta_hv
    p_o = o + t0 * stride_o_t + i_hv * stride_o_hv + i_v * stride_o_v

    p_h = h0 + (i_seq * H0_STRIDE_S + i_hv * H0_STRIDE_HV).to(tl.int64) + offs * V + i_v
    h = tl.load(p_h).to(tl.float32)

    # S_t = exp(g_t) * S_{t-1} + beta_t * k_t^T (v_t - k_t @ S_{t-1})
    # o_t = q_t @ S_t * scale
    for it in range(0, t1 - t0):
        bq = tl.load(p_q).to(tl.float32)
        bk = tl.load(p_k).to(tl.float32)
        bv = tl.load(p_v).to(tl.float32)
        bg = tl.load(p_g).to(tl.float32)
        bb = tl.load(p_beta).to(tl.float32)

        h = h * tl.exp(bg)
        acc = tl.sum(h * bk)  # (v_t - k_t @ S) component
        bv = (bv - acc) * bb
        h = h + bk * bv
        bo = tl.sum(h * bq) * scale
        tl.store(p_o, bo.to(o.dtype.element_ty))

        p_q += stride_q_t
        p_k += stride_k_t
        p_v += stride_v_t
        p_g += stride_g_t
        p_beta += stride_beta_t
        p_o += stride_o_t

    if STORE_FINAL:
        # scalar stores only (vector stores are unreliable on this backend)
        for kk in tl.static_range(0, BK):
            h_kk = tl.sum(tl.where(offs == kk, h, 0.0))
            p_ht = (
                ht
                + (i_seq * HT_STRIDE_S + i_hv * HT_STRIDE_HV).to(tl.int64)
                + kk * V
                + i_v
            )
            tl.store(p_ht, h_kk.to(ht.dtype.element_ty))


def _chunk_size_for_sequence(T: int, is_varlen: bool) -> int:
    # mirrors flag_gems.fused.FLA.chunk._chunk_size_for_sequence
    if is_varlen:
        return 64
    return min(64, max(16, 1 << (T - 1).bit_length()))


@triton.jit
def _chunk_cumsum_kernel(
    g,
    g_out,
    T,  # sequence length (non-varlen)
    cu_seqlens,
    stride_cu,
    chunk: tl.constexpr,
    H: tl.constexpr,
    USE_CU: tl.constexpr,
):
    """Within-chunk cumsum of g along the sequence dim (reset every chunk).

    Mirrors the generic chunk-local cumsum.  Program (i_chunk, i_bh):
    g_out[bos + i_chunk*chunk + i, h] = sum_{j<=i} g[bos + i_chunk*chunk + j, h].

    Uses scalar loads/stores with a runtime-bound loop (the proven pattern of
    the main kernel; small-vector loads/stores miscompile on this backend).
    """
    i_chunk, i_bh = tl.program_id(0), tl.program_id(1)
    i_h = i_bh % H
    if USE_CU:
        i_seq = i_bh // H
        bos = tl.load(cu_seqlens + i_seq * stride_cu).to(tl.int32)
        eos = tl.load(cu_seqlens + (i_seq + 1) * stride_cu).to(tl.int32)
    else:
        bos = i_bh // H * T
        eos = bos + T
    t0 = bos + i_chunk * chunk
    n = eos - t0
    if n > chunk:
        n = chunk
    p = g + t0 * H + i_h
    p_o = g_out + t0 * H + i_h
    acc = tl.zeros([], dtype=tl.float32)
    for i in range(0, n):
        b_g = tl.load(p).to(tl.float32)
        acc += b_g
        tl.store(p_o, acc.to(g_out.dtype.element_ty))
        p += H
        p_o += H


def _chunk_cumsum(g: torch.Tensor, chunk_size: int, cu_seqlens) -> torch.Tensor:
    B, T, H = g.shape
    if cu_seqlens is None:
        N = B
        nchunks = triton.cdiv(T, chunk_size)
    else:
        N = len(cu_seqlens) - 1
        nchunks = triton.cdiv(int(cu_seqlens[-1] - cu_seqlens[0]), chunk_size)
    g_out = torch.empty_like(g)
    _chunk_cumsum_kernel[(nchunks, N * H)](
        g=g,
        g_out=g_out,
        T=T,
        cu_seqlens=cu_seqlens,
        stride_cu=cu_seqlens.stride(0) if cu_seqlens is not None else 0,
        chunk=chunk_size,
        H=H,
        USE_CU=cu_seqlens is not None,
    )
    return g_out


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
    logger.debug("GEMS_KUNLUNXIN CHUNK GATED DELTA RULE FWD")

    # contiguity (mirror the generic wrapper)
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if not g.is_contiguous():
        g = g.contiguous()
    if not beta.is_contiguous():
        beta = beta.contiguous()
    if initial_state is not None and not initial_state.is_contiguous():
        initial_state = initial_state.contiguous()
    if cu_seqlens is not None and not cu_seqlens.is_contiguous():
        cu_seqlens = cu_seqlens.contiguous()

    B, T, H, K = q.shape  # H = q/k heads (Hg)
    HV, V = v.shape[2], v.shape[3]
    if (K & (K - 1)) != 0:
        raise ValueError(
            f"chunk_gated_delta_rule_fwd (kunlunxin) requires power-of-2 K, got {K}"
        )
    if HV % H != 0:
        raise ValueError(
            f"chunk_gated_delta_rule_fwd (kunlunxin) requires H to divide HV, "
            f"got H={H}, HV={HV}"
        )

    chunk_size = _chunk_size_for_sequence(T, cu_seqlens is not None)

    # initial state: (B, H, K, V) for the packed case; for varlen each sequence
    # is assigned the initial state of its (replicated) batch entry.
    if cu_seqlens is None:
        N = B
        h0 = initial_state
    else:
        N = len(cu_seqlens) - 1
        if initial_state is None:
            h0 = None
        elif initial_state.shape[0] == N:
            h0 = initial_state
        else:
            idx = torch.arange(N, device=initial_state.device, dtype=torch.long)
            h0 = initial_state[idx % initial_state.shape[0]]
    if h0 is None:
        h0 = torch.zeros(N, HV, K, V, device=q.device, dtype=torch.float32)

    o = torch.empty(B, T, HV, V, device=q.device, dtype=v.dtype)
    if output_final_state:
        final_state = torch.empty(N, HV, K, V, device=q.device, dtype=torch.float32)
    else:
        final_state = None

    BK = triton.next_power_of_2(K)
    grid = (N, HV, V)
    _chunk_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=h0,
        ht=final_state if final_state is not None else o,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        stride_q_t=q.stride(1),
        stride_q_h=q.stride(2),
        stride_q_k=q.stride(3),
        stride_k_t=k.stride(1),
        stride_k_h=k.stride(2),
        stride_k_k=k.stride(3),
        stride_v_t=v.stride(1),
        stride_v_hv=v.stride(2),
        stride_v_v=v.stride(3),
        stride_o_t=o.stride(1),
        stride_o_hv=o.stride(2),
        stride_o_v=o.stride(3),
        stride_g_t=g.stride(1),
        stride_g_hv=g.stride(2),
        stride_beta_t=beta.stride(1),
        stride_beta_hv=beta.stride(2),
        stride_cu=cu_seqlens.stride(0) if cu_seqlens is not None else 0,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        H0_STRIDE_S=h0.stride(0),
        H0_STRIDE_HV=h0.stride(1),
        HT_STRIDE_S=(final_state.stride(0) if final_state is not None else o.stride(0)),
        HT_STRIDE_HV=(
            final_state.stride(1) if final_state is not None else o.stride(1)
        ),
        USE_CU=cu_seqlens is not None,
        STORE_FINAL=final_state is not None,
    )

    g_out = _chunk_cumsum(g, chunk_size, cu_seqlens)
    A = torch.zeros(B, T, HV, chunk_size, device=q.device, dtype=torch.float32)
    return g_out, o, A, final_state, None, None, None
